# main.py
import os
import asyncio
import logging
import signal
import time
from aiohttp import web
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("bot")

# ── Your existing handlers & jobs ─────────────────────────────────────────────
# Keep your current handlers and your async job here. Example placeholders:

async def start_cmd(update: Update, _ctx):
    await update.message.reply_text("👍 Bot is alive.")

# IMPORTANT: keep YOUR original implementation of this function.
# It already exists in your repo per the logs; reuse it as-is.
async def job_push_alerts(application: Application):
    """
    Your original periodic-alerts coroutine. Implementation unchanged.
    """
    # ... your existing logic ...
    return


# ── Webhook HTTP server (aiohttp) ─────────────────────────────────────────────
def make_web_app(application: Application) -> web.Application:
    secret_path = os.getenv("WEBHOOK_SECRET", "hook")      # path part
    secret_token = os.getenv("WEBHOOK_TOKEN", "")          # header check (optional)

    app = web.Application()

    async def root(_req):
        return web.Response(text="ok")

    async def health(_req):
        return web.json_response({
            "ok": True,
            "uptime_s": int(time.monotonic()),
            "bot_username": application.bot.username if application.bot else None,
        })

    async def webhook(request: web.Request):
        # Optional header verification if you set secret_token during setWebhook
        if secret_token:
            hdr = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if hdr != secret_token:
                return web.Response(status=401, text="unauthorized")
        try:
            data = await request.json()
        except Exception:
            return web.Response(status=400, text="invalid json")

        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(text="ok")

    app.router.add_get("/", root)
    app.router.add_get("/health", health)
    app.router.add_post(f"/webhook/{secret_path}", webhook)
    return app


# ── Startup / shutdown orchestration ──────────────────────────────────────────
async def start_services(application: Application):
    # 1) Initialize PTB (no polling)
    await application.initialize()
    await application.start()

    # 2) Set webhook with Telegram (so Telegram POSTs to our Koyeb URL)
    public_url = os.environ["PUBLIC_URL"]  # e.g., https://your-app.koyeb.app
    secret_path = os.getenv("WEBHOOK_SECRET", "hook")
    secret_token = os.getenv("WEBHOOK_TOKEN", "") or None

    await application.bot.delete_webhook(drop_pending_updates=True)
    await application.bot.set_webhook(
        url=f"{public_url}/webhook/{secret_path}",
        secret_token=secret_token,          # validates header if set
        allowed_updates=Update.ALL_TYPES,   # receive everything
    )
    log.info("Webhook set to %s/webhook/%s", public_url, secret_path)

    # 3) Schedule periodic jobs with PTB JobQueue (loop-safe)
    #    Runs every 15 minutes, starting 30s after boot.
    application.job_queue.run_repeating(
        lambda ctx: job_push_alerts(ctx.application),
        interval=15 * 60,
        first=30,
        name="push_alerts",
    )
    log.info("Scheduled job_push_alerts every 15 minutes.")

    # 4) Start aiohttp web server (health + webhook)
    web_app = make_web_app(application)
    runner = web.AppRunner(web_app)
    await runner.setup()
    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info("HTTP server listening on 0.0.0.0:%s", port)
    return runner


async def stop_services(application: Application, runner: web.AppRunner):
    try:
        await application.bot.delete_webhook()  # optional
    except Exception:
        log.exception("delete_webhook failed (ignored).")
    await application.stop()
    await application.shutdown()
    await runner.cleanup()


async def main():
    # ── Build PTB application from env ──
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    application = Application.builder().token(token).build()

    # ── Register your handlers (add yours below) ──
    application.add_handler(CommandHandler("start", start_cmd))
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, your_handler))

    # ── Start everything ──
    runner = await start_services(application)

    # ── Wait for SIGINT/SIGTERM ──
    loop = asyncio.get_running_loop()
    stop_fut = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_fut.cancel)

    try:
        await stop_fut
    except asyncio.CancelledError:
        pass
    finally:
        await stop_services(application, runner)


if __name__ == "__main__":
    asyncio.run(main())
