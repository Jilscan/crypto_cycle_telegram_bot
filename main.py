
import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Dict, Any, List, Optional, Tuple

import httpx
import yaml
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

# -----------------------------
# Config / Models
# -----------------------------

class Config(BaseModel):
    telegram: Dict[str, Any]
    schedule: Dict[str, Any]
    symbols: Dict[str, List[str]]
    thresholds: Dict[str, Any]  # kept for backward-compat; not used when risk_profiles present
    glassnode: Dict[str, Any] = {}
    force_chat_id: str = ""
    logging: Dict[str, Any] = {"level": "INFO"}
    default_profile: str = "moderate"
    risk_profiles: Dict[str, Dict[str, Any]] | None = None

def load_config(path: str = "config.yml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)

CFG = load_config()


# Per-chat risk profile storage
from pathlib import Path
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
    # If risk_profiles defined, choose active; else fall back to CFG.thresholds
    if CFG.risk_profiles:
        profile = CFG.default_profile
        if chat_id is not None and chat_id in CHAT_PROFILE:
            profile = CHAT_PROFILE[chat_id]
        prof = CFG.risk_profiles.get(profile, CFG.risk_profiles.get(CFG.default_profile))
        if prof and isinstance(prof, dict):
            return prof.get("thresholds", CFG.thresholds)
    return CFG.thresholds

# Logging
logging.basicConfig(level=getattr(logging, CFG.logging.get("level", "INFO")))
log = logging.getLogger("crypto-cycle-bot")

# -----------------------------
# HTTP utils
# -----------------------------

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

# -----------------------------
# Data Providers
# -----------------------------

class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()

    # CoinGecko global data
    async def coingecko_global(self) -> Dict[str, Any]:
        url = "https://api.coingecko.com/api/v3/global"
        return await fetch_json(self.client, url)

    # Binance: ETHBTC price
    async def binance_price(self, symbol: str) -> float:
        url = "https://api.binance.com/api/v3/ticker/price"
        data = await fetch_json(self.client, url, params={"symbol": symbol})
        return float(data["price"])

    # Binance Perp premiumIndex includes funding rate, mark price
    async def binance_premium_index(self, symbol: str) -> Dict[str, Any]:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        return await fetch_json(self.client, url, params={"symbol": symbol})

    # Binance open interest
    async def binance_open_interest(self, symbol: str) -> Dict[str, Any]:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        return await fetch_json(self.client, url, params={"symbol": symbol})

    # Google Trends via pytrends (run in a thread to avoid blocking)
    async def google_trends_score(self, keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        from pytrends.request import TrendReq
        import pandas as pd

        def _work():
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload(keywords, cat=0, timeframe="now 7-d", geo="", gprop="")
            df = pytrends.interest_over_time()
            if df.empty:
                return 0.0, {k: 0.0 for k in keywords}
            # average last 7d scores, dropping isPartial
            out = {}
            for k in keywords:
                out[k] = float(df[k].mean())
            avg = float(pd.Series(out).mean())
            return avg, out

        return await asyncio.to_thread(_work)

    # Glassnode optional endpoints (require API key)
    async def glassnode_mvrv_z(self, asset: str = "BTC") -> Optional[float]:
        # Daily metric; take the latest value
        api_key = CFG.glassnode.get("api_key") or ""
        if not api_key or not CFG.glassnode.get("enable"):
            return None
        url = "https://api.glassnode.com/v1/metrics/market/mvrv_z_score"
        data = await fetch_json(self.client, url, params={"a": asset, "api_key": api_key, "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

    async def glassnode_exchange_netflow(self, asset: str = "BTC") -> Optional[float]:
        api_key = CFG.glassnode.get("api_key") or ""
        if not api_key or not CFG.glassnode.get("enable"):
            return None
        url = "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_adjusted_sum"
        # Netflow is typically inflow - outflow; many endpoints exist. We use adjusted inflow as a rough proxy.
        data = await fetch_json(self.client, url, params={"a": asset, "api_key": api_key, "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

# -----------------------------
# Metrics computation
# -----------------------------

async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    # Parallel fetches
    tasks = []
    tasks.append(dc.coingecko_global())
    tasks.append(dc.binance_price("ETHBTC"))
    for sym in CFG.symbols["funding"]:
        tasks.append(dc.binance_premium_index(sym))
    for sym in CFG.symbols["oi"]:
        tasks.append(dc.binance_open_interest(sym))
    tasks.append(dc.google_trends_score(["crypto", "bitcoin", "ethereum"]))
    tasks.append(dc.glassnode_mvrv_z("BTC"))
    tasks.append(dc.glassnode_exchange_netflow("BTC"))

    results = await asyncio.gather(*tasks)

    # Unpack
    idx = 0
    cg = results[idx]; idx += 1
    ethbtc = results[idx]; idx += 1

    funding = {}
    for sym in CFG.symbols["funding"]:
        funding[sym] = results[idx]; idx += 1

    oi = {}
    for sym in CFG.symbols["oi"]:
        oi[sym] = results[idx]; idx += 1

    trends_avg, trends_by_kw = results[idx]; idx += 1
    mvrv_z = results[idx]; idx += 1
    exch_inflow = results[idx]; idx += 1

    # CoinGecko global structure
    total_mcap = cg["data"]["total_market_cap"].get("usd", 0.0)
    btc_pct = cg["data"]["market_cap_percentage"].get("btc", 0.0)
    eth_pct = cg["data"]["market_cap_percentage"].get("eth", 0.0)
    btc_mcap = total_mcap * (btc_pct/100.0)
    eth_mcap = total_mcap * (eth_pct/100.0)
    altcap = max(total_mcap - btc_mcap - eth_mcap, 0.0)
    altcap_btc_ratio = (altcap / btc_mcap) if btc_mcap > 0 else 0.0

    # Compute OI notional USD using mark price (from premiumIndex) if needed; here we only have raw OI coin qty endpoint.
    # For simplicity, we fetch BTC/ETH mark price from premiumIndex already in 'funding' dict.
    def _oi_usd(symbol: str) -> Optional[float]:
        try:
            raw = oi[symbol]
            qty = float(raw.get("openInterest", 0.0))  # in base asset
            # find mark price for that symbol from 'funding' dict (premiumIndex includes markPrice)
            prem = funding.get(symbol, {})
            mark = float(prem.get("markPrice", 0.0))
            return qty * mark
        except Exception:
            return None

    oi_usd = {sym: _oi_usd(sym) for sym in CFG.symbols["oi"]}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_dominance_pct": btc_pct,
        "eth_btc": ethbtc,
        "altcap_btc_ratio": altcap_btc_ratio,
        "funding": funding,
        "open_interest_usd": oi_usd,
        "google_trends_avg7d": trends_avg,
        "google_trends_breakdown": trends_by_kw,
        "mvrv_z": mvrv_z,
        "exchange_inflow_proxy": exch_inflow,
    }

def build_status_text(metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"📊 *Crypto Cycle Snapshot* (`{metrics['timestamp']}` UTC)")
    lines.append(f"• BTC Dominance: *{metrics['btc_dominance_pct']:.2f}%*")
    lines.append(f"• ETH/BTC: *{metrics['eth_btc']:.5f}*")
    lines.append(f"• Altcap/BTC: *{metrics['altcap_btc_ratio']:.2f}*")

    # Funding
    lines.append("• Funding (8h, est):")
    for sym, d in metrics["funding"].items():
        fr = float(d.get("lastFundingRate", 0.0)) * 100  # to %
        lines.append(f"   - {sym}: *{fr:.3f}%*")

    # OI
    lines.append("• Open Interest (USD est):")
    for sym, notional in metrics["open_interest_usd"].items():
        if notional is not None:
            lines.append(f"   - {sym}: *${notional:,.0f}*")
        else:
            lines.append(f"   - {sym}: n/a")

    # Trends
    lines.append(f"• Google Trends 7d avg: *{metrics['google_trends_avg7d']:.1f}* " +
                 f"(crypto {metrics['google_trends_breakdown']['crypto']:.1f}, " +
                 f"bitcoin {metrics['google_trends_breakdown']['bitcoin']:.1f}, " +
                 f"ethereum {metrics['google_trends_breakdown']['ethereum']:.1f})")

    if metrics.get("mvrv_z") is not None:
        lines.append(f"• BTC MVRV Z-Score (Glassnode): *{metrics['mvrv_z']:.2f}*")
    if metrics.get("exchange_inflow_proxy") is not None:
        lines.append(f"• Exchange inflow proxy (Glassnode, adj sum): *{metrics['exchange_inflow_proxy']:.2f}*")

    return "\n".join(lines)

def evaluate_flags(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[int, List[str]]:
    t = thresholds
    flags = []

    if metrics["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("BTC dominance high")

    if metrics["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("ETH/BTC elevated")

    if metrics["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Altcap/BTC stretched")

    # funding
    for sym, d in metrics["funding"].items():
        fr = abs(float(d.get("lastFundingRate", 0.0)) * 100)  # %
        if fr >= t["funding_rate_abs_max"]:
            flags.append(f"Funding extreme: {sym} ({fr:.3f}%)")
            break

    # OI
    oi_usd = metrics["open_interest_usd"]
    btc_oi = oi_usd.get("BTCUSDT")
    eth_oi = oi_usd.get("ETHUSDT")
    if btc_oi and btc_oi >= t["open_interest_btc_usdt_usd_max"]:
        flags.append(f"BTC OI high (${btc_oi:,.0f})")
    if eth_oi and eth_oi >= t["open_interest_eth_usdt_usd_max"]:
        flags.append(f"ETH OI high (${eth_oi:,.0f})")

    # Trends
    if metrics["google_trends_avg7d"] >= t["google_trends_7d_avg_min"]:
        flags.append("Retail interest (Google) high")

    # Optional Glassnode
    if metrics.get("mvrv_z") is not None and metrics["mvrv_z"] >= 7.0:
        flags.append("MVRV Z-Score extreme")

    return len(flags), flags

# -----------------------------
# Telegram Bot
# -----------------------------

async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not CFG.risk_profiles:
        await update.message.reply_text("Risk profiles are not configured. Using static thresholds.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /risk <conservative|moderate|aggressive>")
        return
    choice = context.args[0].lower()
    if choice not in CFG.risk_profiles:
        opts = ", ".join(CFG.risk_profiles.keys())
        await update.message.reply_text(f"Unknown profile '{choice}'. Options: {opts}")
        return
    CHAT_PROFILE[chat_id] = choice
    save_profiles()
    await update.message.reply_text(f"✅ Risk profile set to *{choice}* for this chat.", parse_mode="Markdown")

async def cmd_getrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if not CFG.risk_profiles:
        await update.message.reply_text("Using static thresholds (no profiles configured).")
        return
    cur = CHAT_PROFILE.get(chat_id, CFG.default_profile)
    await update.message.reply_text(f"Current risk profile for this chat: *{cur}*", parse_mode="Markdown")


SUBSCRIBERS: set[int] = set()

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    await update.message.reply_text(
        "👋 Crypto Cycle Watch online.\n"
        "Use /status for a snapshot, /subscribe to get alerts, "
        "/settime HH:MM to set daily summary time, /unsubscribe to stop."
    )

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dc = DataClient()
    try:
        metrics = await gather_metrics(dc)
        text = build_status_text(metrics)
        # annotate with profile
        if CFG.risk_profiles:
            cur = CHAT_PROFILE.get(update.effective_chat.id, CFG.default_profile)
            text += f"\n• Risk profile: *{cur}*"
        t = get_thresholds_for_chat(update.effective_chat.id if 'update' in locals() and update and update.effective_chat else None)
        nflags, flags = evaluate_flags(metrics, t)
        if nflags > 0:
            text += "\n\n⚠️ Flags: " + ", ".join(flags)
        await update.message.reply_markdown(text)
    finally:
        await dc.close()

async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    await update.message.reply_text("✅ Subscribed. You'll receive alerts and daily summaries here.")

async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("🛑 Unsubscribed.")

async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args[0]) != 5 or context.args[0][2] != ":":
        await update.message.reply_text("Usage: /settime HH:MM (24h)")
        return
    CFG.schedule["daily_summary_time"] = context.args[0]
    await update.message.reply_text(f"🕒 Daily summary time set to {context.args[0]} (server time).")

async def push_summary(app: Application):
    dc = DataClient()
    try:
        metrics = await gather_metrics(dc)
        text = build_status_text(metrics)
        # annotate with profile
        if CFG.risk_profiles:
            cur = CHAT_PROFILE.get(update.effective_chat.id, CFG.default_profile)
            text += f"\n• Risk profile: *{cur}*"
        t = get_thresholds_for_chat(update.effective_chat.id if 'update' in locals() and update and update.effective_chat else None)
        nflags, flags = evaluate_flags(metrics, t)
        if nflags > 0:
            text += "\n\n⚠️ Flags: " + ", ".join(flags)
        targets = set(SUBSCRIBERS)
        if CFG.force_chat_id:
            try:
                targets.add(int(CFG.force_chat_id))
            except Exception:
                pass
        for chat_id in targets:
            try:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
            except Exception as e:
                log.warning("Failed to send summary to %s: %s", chat_id, e)
    finally:
        await dc.close()

async def push_alerts(app: Application):
    dc = DataClient()
    try:
        metrics = await gather_metrics(dc)
        t = get_thresholds_for_chat(update.effective_chat.id if 'update' in locals() and update and update.effective_chat else None)
        nflags, flags = evaluate_flags(metrics, t)
        if nflags >= CFG.thresholds["min_flags_for_alert"]:
            text = "🚨 *Top Risk Alert*\n" + build_status_text(metrics) + "\n\n⚠️ Flags: " + ", ".join(flags)
            targets = set(SUBSCRIBERS)
            if CFG.force_chat_id:
                try:
                    targets.add(int(CFG.force_chat_id))
                except Exception:
                    pass
            for chat_id in targets:
                try:
                    await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
                except Exception as e:
                    log.warning("Failed to send alert to %s: %s", chat_id, e)
    finally:
        await dc.close()

def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

async def main():
    token = CFG.telegram["token"]
    app: Application = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("risk", cmd_risk))
    app.add_handler(CommandHandler("getrisk", cmd_getrisk))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("settime", cmd_settime))

    # Load saved chat profiles
    load_profiles()

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=str(datetime.now().astimezone().tzinfo))
    # Daily summary
    hh, mm = parse_hhmm(CFG.schedule.get("daily_summary_time", "13:00"))
    scheduler.add_job(lambda: asyncio.create_task(push_summary(app)),
                      CronTrigger(hour=hh, minute=mm))
    # Alerts every 15 minutes
    scheduler.add_job(lambda: asyncio.create_task(push_alerts(app)),
                      CronTrigger(minute="*/15"))
    scheduler.start()

    # Warm-up: announce if force_chat_id is set
    if CFG.force_chat_id:
        try:
            await app.bot.send_message(chat_id=int(CFG.force_chat_id),
                                       text="🤖 Crypto Cycle Watch bot started.")
        except Exception as e:
            log.warning("Unable to notify force_chat_id: %s", e)

    log.info("Bot running. Press Ctrl+C to exit.")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
