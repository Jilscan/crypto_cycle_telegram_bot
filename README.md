
# Crypto Cycle Watch — Telegram Bot (Fixed for Koyeb Web Service)

This build fixes:
- APScheduler timezone error (uses IANA TZ from `TZ` env var).
- Python-Telegram-Bot event-loop conflict (uses `initialize()`/`start()` instead of `run_polling()`).
- Adds a tiny HTTP server with `aiohttp` to answer `/health` for Koyeb health checks.

## Deploy (Koyeb Web Service, Free plan)

1) Set env vars in Koyeb Service → Settings → Environment Variables:
   - `TELEGRAM_TOKEN` = <your BotFather token>
   - `TZ` = `America/New_York` (or `UTC`)

2) Service type: **Web Service**
   - Build method: **Dockerfile**
   - Expose a port (e.g., 8080)
   - Health check: HTTP path `/health`

3) Redeploy. Logs should show:
   - "Scheduler started"
   - "Bot running. Press Ctrl+C to exit."

4) In Telegram, open your bot → **Start** → `/status`.

## Commands
- `/start`, `/status`, `/subscribe`, `/unsubscribe`
- `/settime HH:MM` (daily summary time, server local)
- `/risk <conservative|moderate|aggressive>`
- `/getrisk`
