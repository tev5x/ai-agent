import os
import sys
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

THRESHOLD = 94.0
STATE_FILE = "state/last_alert_date.txt"

CURRENCY_URL = "https://open.er-api.com/v6/latest/USD"

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]


def log(message):
    print(f"[Rupee Sentinel AI] {message}")


def get_env(name, required=True):
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"Missing required secret/environment variable: {name}")
    return value


def today_ist():
    return datetime.now(IST).date().isoformat()


def current_time_ist_label():
    return datetime.now(IST).strftime("%I:%M %p").lstrip("0")


def already_alerted_today():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            last_date = f.read().strip()
        return last_date == today_ist()
    except FileNotFoundError:
        return False


def mark_alerted_today():
    os.makedirs("state", exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.write(today_ist())


def fetch_usd_inr_rate():
    response = requests.get(CURRENCY_URL, timeout=20)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise RuntimeError(f"Currency API did not return success: {data}")

    rates = data.get("rates", {})
    if "INR" not in rates:
        raise RuntimeError("Currency API response does not contain INR rate.")

    rate = float(rates["INR"])

    update_unix = data.get("time_last_update_unix")
    update_time = None
    if update_unix:
        update_time = datetime.fromtimestamp(int(update_unix), timezone.utc)

    return rate, update_time


def data_is_usable(update_time):
    """
    Internal kitchen check.
    Do not expose this to the customer.
    Open exchange-rate source updates daily, so we allow up to 3 days
    to avoid weekend/holiday false rejection.
    """
    if update_time is None:
        return True

    now_utc = datetime.now(timezone.utc)
    age_days = (now_utc - update_time).days

    return age_days <= 3


def default_clean_message(rate, checked_at):
    return (
        "🚨 Rupee Sentinel AI\n\n"
        "USD/INR is below ₹94.\n\n"
        f"Current rate: ₹{rate:.2f}\n"
        f"Checked at: {checked_at} IST"
    )


def create_ai_message_with_gemini(rate, checked_at):
    """
    AI layer.
    Gemini's job is only to produce the customer-ready dosa.
    No confidence, no explanation, no kitchen smoke.
    """
    gemini_api_key = get_env("GEMINI_API_KEY")

    prompt = f"""
Create a short Telegram alert.

Rules:
- Output only the final customer-facing message.
- No confidence score.
- No explanation.
- No data source details.
- No advice.
- No markdown.
- No extra text before or after.
- Keep exactly this meaning.

Message must say:
Rupee Sentinel AI alert.
USD/INR is below ₹94.
Current rate is ₹{rate:.2f}.
Checked at {checked_at} IST.

Preferred format:
🚨 Rupee Sentinel AI

USD/INR is below ₹94.

Current rate: ₹{rate:.2f}
Checked at: {checked_at} IST
""".strip()

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 120,
        },
    }

    last_error = None

    for model in GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={gemini_api_key}"
        )

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            text = (
                data["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
            )

            if "USD/INR is below ₹94" in text and "Current rate:" in text:
                return text

            last_error = f"AI output failed format check: {text}"

        except Exception as e:
            last_error = str(e)
            continue

    log(f"AI composer failed. Using safety fallback. Reason: {last_error}")
    return default_clean_message(rate, checked_at)


def send_telegram_message(text):
    telegram_bot_token = get_env("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = get_env("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"

    payload = {
        "chat_id": telegram_chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    response = requests.post(url, json=payload, timeout=20)
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram send failed: {data}")

    return data


def main():
    force_alert = os.getenv("FORCE_ALERT", "false").lower() == "true"

    log("Started.")

    if already_alerted_today() and not force_alert:
        log("Already alerted today. Staying silent.")
        return

    rate, update_time = fetch_usd_inr_rate()
    checked_at = current_time_ist_label()

    log(f"Fetched USD/INR rate: {rate:.4f}")

    if not data_is_usable(update_time):
        log("Currency data failed freshness check. Staying silent.")
        return

    if rate >= THRESHOLD and not force_alert:
        log(f"Rate is not below ₹{THRESHOLD:.2f}. Staying silent.")
        return

    message = create_ai_message_with_gemini(rate, checked_at)

    send_telegram_message(message)

    mark_alerted_today()

    log("Alert sent and daily state updated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Failed: {e}")
        sys.exit(1)
