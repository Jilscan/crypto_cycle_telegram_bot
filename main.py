from pathlib import Path
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import httpx
import yaml
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

# ───────────────────────────
# CONFIG (file with fallback)
# ───────────────────────────

DEFAULTS: Dict[str, Any] = {
    "telegram": {"token": ""},  # we prefer TELEGRAM_TOKEN env
    "schedule": {"daily_summary_time": "13:00"},
    "symbols": {
        "funding": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
        "oi": ["BTCUSDT", "ETHUSDT"],
    },
    "thresholds": {
        "min_flags_for_alert": 3,
        "btc_dominance_max": 60.0,
        "eth_btc_ratio_max": 0.09,
        "altcap_btc_ratio_max": 1.8,
        "funding_rate_abs_max": 0.10,
        "open_interest_btc_usdt_usd_max": 20_000_000_000,
        "open_interest_eth_usdt_usd_max": 8_000_000_000,
        "google_trends_7d_avg_min": 75,
        "mvrv_z_extreme": 7.0,
    },
    "glassnode": {"api_key": "", "enable": False},
    "force_chat_id": "",
    "logging": {"level": "INFO"},
    "default_profile": "moderate",
    "risk_profiles": {
        "conservative": {"thresholds": {
            "min_flags_for_alert": 2, "btc_dominance_max": 57.0, "eth_btc_ratio_max": 0.085,
            "altcap_btc_ratio_max": 1.6, "funding_rate_abs_max": 0.06,
            "open_interest_btc_usdt_usd_max": 15_000_000_000,
            "open_interest_eth_usdt_usd_max": 6_000_000_000, "google_trends_7d_avg_min": 60,
            "mvrv_z_extreme": 6.5}},
        "moderate": {"thresholds": {
            "min_flags_for_alert": 3, "btc_dominance_max": 60.0, "eth_btc_ratio_max": 0.09,
            "altcap_btc_ratio_max": 1.8, "funding_rate_abs_max": 0.10,
            "open_interest_btc_usdt_usd_max": 20_000_000_000,
            "open_interest_eth_usdt_usd_max": 8_000_000_000, "google_trends_7d_avg_min": 75,
            "mvrv_z_extreme": 7.0}},
        "aggressive": {"thresholds": {
            "min_flags_for_alert": 4, "btc_dominance_max": 62.0, "eth_btc_ratio_max": 0.095,
            "altcap_btc_ratio_max": 2.0, "funding_rate_abs_max": 0.14,
            "open_interest_btc_usdt_usd_max": 25_000_000_000,
            "open_interest_eth_usdt_usd_max": 10_000_000_000, "google_trends_7d_avg_min": 85,
            "mvrv_z_extreme": 8.0}},
    },
}


class Config(BaseModel):
    telegram: Dict[str, Any]
    schedule: Dict[str, Any]
    symbols: Dict[str, List[str]]
    thresholds: Dict[str, Any]
    glassnode: Dict[str, Any] = {}
    force_chat_id: str = ""
    logging: Dict[str, Any] = {"level": "INFO"}
    default_profile: str = "moderate"
    risk_profiles: Dict[str, Dict[str, Any]] | None = None


def load_config(path: str = "config.yml") -> "Config":
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        # overlay defaults to fill any missing keys

        def deep_merge(base, extra):
            if isinstance(base, dict) and isinstance(extra, dict):
                out = dict(base)
                for k, v in extra.items():
                    out[k] = deep_merge(base.get(k), v)
                return out
            return extra if extra is not None else base
        merged = deep_merge(DEFAULTS, raw)
        src = "config.yml"
    except FileNotFoundError:
        merged = DEFAULTS
        src = "(built-in defaults)"
    cfg = Config(**merged)
    logging.basicConfig(level=getattr(
        logging, cfg.logging.get("level", "INFO")))
    logging.getLogger(
        "crypto-cycle-bot").info(f"Loaded configuration from {src}")
    return cfg


CFG = load_config()

# per-chat risk profile persistence
PROFILE_FILE = Path("profiles.yml")
CHAT_PROFILE: dict[int, str] = {}


def load_profiles():
    global CHAT_PROFILE
    if PROFILE_FILE.exists():
        try:
            CHAT_PROFILE = yaml.safe_load(PROFILE_FILE.read_text()) or {}
        except Exception:
            CHAT_PROFILE = {}
    else:
        CHAT_PROFILE = {}


def save_profiles():
    try:
        PROFILE_FILE.write_text(yaml.safe_dump(CHAT_PROFILE, sort_keys=False))
    except Exception as e:
        log.warning("Failed to save profiles.yml: %s", e)


def get_thresholds_for_chat(chat_id: Optional[int]) -> Dict[str, Any]:
    if CFG.risk_profiles:
        profile = CHAT_PROFILE.get(
            chat_id, CFG.default_profile) if chat_id else CFG.default_profile
        prof = CFG.risk_profiles.get(
            profile, CFG.risk_profiles.get(CFG.default_profile))
        if prof and isinstance(prof, dict):
            return prof.get("thresholds", CFG.thresholds)
    return CFG.thresholds


# logging
log = logging.getLogger("crypto-cycle-bot")

# ───────────────────────────
# HTTP helpers & data clients
# ───────────────────────────
HEADERS = {"User-Agent": "crypto-cycle-bot/1.0"}


async def fetch_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] | None = None) -> Any:
    for attempt in range(3):
        try:
            r = await client.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("GET %s failed (attempt %d): %s", url, attempt+1, e)
            await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url}")


class DataClient:
    def __init__(self): self.client = httpx.AsyncClient()
    async def close(self): await self.client.aclose()

    async def coingecko_global(self) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://api.coingecko.com/api/v3/global")

    async def binance_price(self, symbol: str) -> float:
        data = await fetch_json(self.client, "https://api.binance.com/api/v3/ticker/price", {"symbol": symbol})
        return float(data["price"])

    async def binance_premium_index(self, symbol: str) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://fapi.binance.com/fapi/v1/premiumIndex", {"symbol": symbol})

    async def binance_open_interest(self, symbol: str) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://fapi.binance.com/fapi/v1/openInterest", {"symbol": symbol})

    async def google_trends_score(self, keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        from pytrends.request import TrendReq
        import pandas as pd

        def _work():
            p = TrendReq(hl="en-US", tz=0)
            p.build_payload(keywords, timeframe="now 7-d")
            df = p.interest_over_time()
            if df.empty:
                return 0.0, {k: 0.0 for k in keywords}
            out = {k: float(df[k].mean()) for k in keywords}
            return float(pd.Series(out).mean()), out
        return await asyncio.to_thread(_work)

    async def glassnode_mvrv_z(self, asset: str = "BTC") -> Optional[float]:
        if not (CFG.glassnode.get("enable") and CFG.glassnode.get("api_key")):
            return None
        data = await fetch_json(self.client, "https://api.glassnode.com/v1/metrics/market/mvrv_z_score",
                                {"a": asset, "api_key": CFG.glassnode["api_key"], "i": "24h"})
        return float(data[-1]["v"]) if isinstance(data, list) and data else None

    async def glassnode_exchange_inflow(self, asset: str = "BTC") -> Optional[float]:
        if not (CFG.glassnode.get("enable") and CFG.glassnode.get("api_key")):
            return None
        data = await fetch_json(self.client, "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_adjusted_sum",
                                {"a": asset, "api_key": CFG.glassnode["api_key"], "i": "24h"})
        return float(data[-1]["v"]) if isinstance(data, list) and data else None

# ───────────────────────────
# Metrics
# ───────────────────────────


async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    tasks = [dc.coingecko_global(), dc.binance_price("ETHBTC")]
    for s in CFG.symbols["funding"]:
        tasks.append(dc.binance_premium_index(s))
    for s in CFG.symbols["oi"]:
        tasks.append(dc.binance_open_interest(s))
    tasks += [dc.google_trends_score(["crypto", "bitcoin", "ethereum"]),
              dc.glassnode_mvrv_z("BTC"), dc.glassnode_exchange_inflow("BTC")]
    results = await asyncio.gather(*tasks)

    i = 0
    cg, ethbtc = results[i], results[i+1]
    i += 2
    funding, oi = {}, {}
    for s in CFG.symbols["funding"]:
        funding[s] = results[i]
        i += 1
    for s in CFG.symbols["oi"]:
        oi[s] = results[i]
        i += 1
    trends_avg, trends_by_kw = results[i]
    i += 1
    mvrv_z, exch_inflow = results[i], results[i+1]

    total = cg["data"]["total_market_cap"].get("usd", 0.0)
    btc_pct = cg["data"]["market_cap_percentage"].get("btc", 0.0)
    eth_pct = cg["data"]["market_cap_percentage"].get("eth", 0.0)
    btc_mcap = total * (btc_pct/100.0)
    eth_mcap = total * (eth_pct/100.0)
    altcap = max(total - btc_mcap - eth_mcap, 0.0)
    alt_btc = (altcap / btc_mcap) if btc_mcap > 0 else 0.0

    def oi_usd(sym: str) -> Optional[float]:
        try:
            qty = float(oi[sym].get("openInterest", 0.0))
            mark = float(funding.get(sym, {}).get("markPrice", 0.0))
            return qty * mark
        except Exception:
            return None

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_dominance_pct": btc_pct, "eth_btc": ethbtc, "altcap_btc_ratio": alt_btc,
        "funding": funding, "open_interest_usd": {s: oi_usd(s) for s in CFG.symbols["oi"]},
        "google_trends_avg7d": trends_avg, "google_trends_breakdown": trends_by_kw,
        "mvrv_z": mvrv_z, "exchange_inflow_proxy": exch_inflow,
    }


def build_status_text(m: Dict[str, Any]) -> str:
    lines = [
        f"📊 *Crypto Cycle Snapshot* (`{m['timestamp']}` UTC)",
        f"• BTC Dominance: *{m['btc_dominance_pct']:.2f}%*",
        f"• ETH/BTC: *{m['eth_btc']:.5f}*",
        f"• Altcap/BTC: *{m['altcap_btc_ratio']:.2f}*",
        "• Funding (8h, est):",
    ]
    for sym, d in m["funding"].items():
        fr = float(d.get("lastFundingRate", 0.0)) * 100
        lines.append(f"   - {sym}: *{fr:.3f}%*")
    lines.append("• Open Interest (USD est):")
    for sym, notional in m["open_interest_usd"].items():
        lines.append(
            f"   - {sym}: *${notional:,.0f}*" if notional is not None else f"   - {sym}: n/a")
    lines.append(
        f"• Google Trends 7d avg: *{m['google_trends_avg7d']:.1f}* (crypto {m['google_trends_breakdown']['crypto']:.1f}, "
        f"bitcoin {m['google_trends_breakdown']['bitcoin']:.1f}, ethereum {m['google_trends_breakdown']['ethereum']:.1f})"
    )
    if m.get("mvrv_z") is not None:
        lines.append(f"• BTC MVRV Z-Score (Glassnode): *{m['mvrv_z']:.2f}*")
    if m.get("exchange_inflow_proxy") is not None:
        lines.append(
            f"• Exchange inflow proxy (Glassnode, adj sum): *{m['exchange_inflow_proxy']:.2f}*")
    return "\n".join(lines)


def evaluate_flags(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[str]]:
    flags: List[str] = []
    if m["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("BTC dominance high")
    if m["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("ETH/BTC elevated")
    if m["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Altcap/BTC stretched")
    for sym, d in m["funding"].items():
        fr = abs(float(d.get("lastFundingRate", 0.0)) * 100)
        if fr >= t["funding_rate_abs_max"]:
            flags.append(f"Funding extreme: {sym} ({fr:.3f}%)")
            break
    oi = m["open_interest_usd"]
    if oi.get("BTCUSDT") and oi["BTCUSDT"] >= t["open_interest_btc_usdt_usd_max"]:
        flags.append(f"BTC OI high (${oi['BTCUSDT']:,.0f})")
    if oi.get("ETHUSDT") and oi["ETHUSDT"] >= t["open_interest_eth_usdt_usd_max"]:
        flags.append(f"ETH OI high (${oi['ETHUSDT']:,.0f})")
    if m["google_trends_avg7d"] >= t["google_trends_7d_avg_min"]:
        flags.append("Retail interest (Google) high")
    if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None and m["mvrv_z"] >= t["mvrv_z_extreme"]:
        flags.append("MVRV Z-Score extreme")
    return len(flags), flags

# ───────────────────────────
# Telegram bot commands
# ───────────────────────────


SUBSCRIBERS: set[int] = set()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("👋 Crypto Cycle Watch online.\nUse /status, /subscribe, /risk <...>, /settime HH:MM, /unsubscribe.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        text = build_status_text(m)
        if CFG.risk_profiles:
            cur = CHAT_PROFILE.get(
                update.effective_chat.id, CFG.default_profile)
            text += f"\n• Risk profile: *{cur}*"
        t = get_thresholds_for_chat(update.effective_chat.id)
        n, flags = evaluate_flags(m, t)
        if n > 0:
            text += "\n\n⚠️ Flags: " + ", ".join(flags)
        await update.message.reply_markdown(text)
    finally:
        await dc.close()


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.discard(update.effective_chat.id)
    await update.message.reply_text("🛑 Unsubscribed.")


async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args[0]) != 5 or context.args[0][2] != ":":
        await update.message.reply_text("Usage: /settime HH:MM (24h)")
        return
    CFG.schedule["daily_summary_time"] = context.args[0]
    await update.message.reply_text(f"🕒 Daily summary time set to {context.args[0]} (server time).")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CFG.risk_profiles:
        await update.message.reply_text("Risk profiles not configured; using static thresholds.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /risk <conservative|moderate|aggressive>")
        return
    choice = context.args[0].lower()
    if choice not in CFG.risk_profiles:
        await update.message.reply_text(f"Unknown profile '{choice}'. Options: {', '.join(CFG.risk_profiles.keys())}")
        return
    CHAT_PROFILE[update.effective_chat.id] = choice
    save_profiles()
    await update.message.reply_text(f"✅ Risk profile set to *{choice}*.", parse_mode="Markdown")


async def cmd_getrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cur = CHAT_PROFILE.get(update.effective_chat.id, CFG.default_profile)
    await update.message.reply_text(f"Current risk profile: *{cur}*", parse_mode="Markdown")


async def push_summary(app: Application):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        text = build_status_text(m)
        t = get_thresholds_for_chat(None)
        n, flags = evaluate_flags(m, t)
        if n > 0:
            text += "\n\n⚠️ Flags: " + ", ".join(flags)
        for chat_id in set(SUBSCRIBERS):
            try:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            except Exception as e:
                log.warning("Failed to send summary to %s: %s", chat_id, e)
    finally:
        await dc.close()


async def push_alerts(app: Application):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        t = get_thresholds_for_chat(None)
        n, flags = evaluate_flags(m, t)
        if n >= t.get("min_flags_for_alert", 3):
            text = "🚨 *Top Risk Alert*\n" + \
                build_status_text(m) + "\n\n⚠️ Flags: " + ", ".join(flags)
            for chat_id in set(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                except Exception as e:
                    log.warning("Failed to send alert to %s: %s", chat_id, e)
    finally:
        await dc.close()


def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

# ───────────────────────────
# Health server
# ───────────────────────────


async def start_health_server():
    from aiohttp import web
    port = int(os.getenv("PORT", "8080"))
    app = web.Application()
    async def ping(request): return web.Response(text="ok")
    app.add_routes([web.get("/", ping), web.get("/health", ping)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info(f"Health server listening on :{port}")

# ───────────────────────────
# Main
# ───────────────────────────


async def main():
    # Token: prefer env for Koyeb; fall back to file only if present
    token = os.getenv("TELEGRAM_TOKEN") or CFG.telegram.get("token", "")
    if not token:
        log.error(
            "No TELEGRAM_TOKEN set and no token in config.yml. Set TELEGRAM_TOKEN env.")
        raise SystemExit(1)

    app: Application = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("settime", cmd_settime))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("getrisk", cmd_getrisk))

    load_profiles()

    # Start health server first so health checks pass
    await start_health_server()

    # Scheduler bound to current loop
    tzname = os.getenv("TZ", "UTC")
    loop = asyncio.get_running_loop()
    scheduler = AsyncIOScheduler(timezone=tzname, event_loop=loop)

    hh, mm = parse_hhmm(CFG.schedule.get("daily_summary_time", "13:00"))
    scheduler.add_job(push_summary, CronTrigger(
        hour=hh, minute=mm), args=[app])
    scheduler.add_job(push_alerts, CronTrigger(minute="*/15"), args=[app])
    scheduler.start()

    if CFG.force_chat_id:
        try:
            await app.bot.send_message(chat_id=int(CFG.force_chat_id), text="🤖 Crypto Cycle Watch bot started.")
        except Exception as e:
            log.warning("Unable to notify force_chat_id: %s", e)

    log.info("Bot running. Press Ctrl+C to exit.")

    await app.initialize()
    await app.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
