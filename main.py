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
    "telegram": {"token": ""},  # prefer TELEGRAM_TOKEN env on Koyeb
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
        "funding_rate_abs_max": 0.10,  # percent
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
log = logging.getLogger("crypto-cycle-bot")

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

# ───────────────────────────
# HTTP helpers & data clients
# ───────────────────────────


OKX_BASE = "https://www.okx.com"
HEADERS = {"User-Agent": "crypto-cycle-bot/1.1", "Accept": "application/json"}


def to_okx_instId(sym: str) -> str:
    # "BTCUSDT" -> "BTC-USDT-SWAP"
    if sym.endswith("USDT"):
        base = sym[:-5]
        quote = "USDT"
    elif sym.endswith("USD"):
        base = sym[:-3]
        quote = "USD"
    else:
        base = sym
        quote = "USDT"
    return f"{base}-{quote}-SWAP"


async def fetch_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] | None = None) -> Any:
    for attempt in range(3):
        try:
            r = await client.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("GET %s failed (attempt %d): %s", url, attempt + 1, e)
            await asyncio.sleep(1 + attempt)
    raise RuntimeError(f"Failed to fetch {url}")


async def _safe(coro, default=None, label: str = ""):
    try:
        return await coro
    except Exception as e:
        log.warning("%s failed: %s", label or getattr(
            coro, "__name__", "call"), e)
        return default


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()

    # CoinGecko global (dominance, totals)
    async def coingecko_global(self) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://api.coingecko.com/api/v3/global")

    # ETH/BTC via CoinGecko
    async def coingecko_eth_btc(self) -> float:
        data = await fetch_json(
            self.client,
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
        )
        eth_usd = float(data["ethereum"]["usd"])
        btc_usd = float(data["bitcoin"]["usd"])
        return eth_usd / btc_usd

    # ── OKX v5 public endpoints (no API key needed) ─────────────────────────
    async def okx_ticker_last_price(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/market/ticker", {"instId": instId})
        try:
            lst = data.get("data", [])
            if lst:
                return float(lst[0]["last"])
        except Exception:
            pass
        return None

    async def okx_funding_rate(self, instId: str) -> Optional[float]:
        # current period funding rate
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/funding-rate", {"instId": instId})
        try:
            lst = data.get("data", [])
            if lst:
                # decimal, e.g., 0.001 = 0.1%
                return float(lst[0]["fundingRate"])
        except Exception:
            pass
        return None

    async def okx_open_interest_usd(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/open-interest", {"instId": instId})
        try:
            lst = data.get("data", [])
            if lst:
                oi_usd = lst[0].get("oiUsd")
                if oi_usd is not None:
                    return float(oi_usd)
        except Exception:
            pass
        return None

    # Google Trends (blocking lib → run in a thread)
    async def google_trends_score(self, keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        import pandas as pd  # noqa
        from pytrends.request import TrendReq  # type: ignore

        def _fetch():
            py = TrendReq(hl="en-US", tz=0)
            py.build_payload(kw_list=keywords, timeframe="now 7-d", geo="")
            df = py.interest_over_time()
            if df is None or df.empty:
                return 0.0, {k: 0.0 for k in keywords}
            means = {k: float(pd.to_numeric(
                df[k], errors="coerce").mean()) for k in keywords}
            avg = sum(means.values()) / max(len(means), 1)
            return avg, means

        return await asyncio.to_thread(_fetch)

    # Optional: Glassnode (requires key + enable=true)
    async def glassnode_mvrv_z(self, asset: str = "BTC") -> Optional[float]:
        if not (CFG.glassnode.get("enable") and CFG.glassnode.get("api_key")):
            return None
        data = await fetch_json(self.client,
                                "https://api.glassnode.com/v1/metrics/market/mvrv_z_score",
                                {"a": asset, "api_key": CFG.glassnode["api_key"], "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

    async def glassnode_exchange_inflow(self, asset: str = "BTC") -> Optional[float]:
        if not (CFG.glassnode.get("enable") and CFG.glassnode.get("api_key")):
            return None
        data = await fetch_json(self.client,
                                "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_adjusted_sum",
                                {"a": asset, "api_key": CFG.glassnode["api_key"], "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

# ───────────────────────────
# Metrics (resilient)
# ───────────────────────────


async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    cg = await _safe(
        dc.coingecko_global(),
        {"data": {"total_market_cap": {"usd": 0.0},
                  "market_cap_percentage": {"btc": 0.0, "eth": 0.0}}},
        "coingecko_global"
    )
    ethbtc = await _safe(dc.coingecko_eth_btc(), 0.0, "coingecko_eth_btc")

    # funding + last price (OKX)
    funding: Dict[str, Dict[str, Any]] = {}
    for sym in CFG.symbols["funding"]:
        inst = to_okx_instId(sym)
        fr = await _safe(dc.okx_funding_rate(inst), 0.0, f"funding {inst}")
        lp = await _safe(dc.okx_ticker_last_price(inst), 0.0, f"lastPrice {inst}")
        funding[sym] = {"lastFundingRate": fr or 0.0, "markPrice": lp or 0.0}

    # open interest USD (OKX)
    oi_usd: Dict[str, Optional[float]] = {}
    for sym in CFG.symbols["oi"]:
        inst = to_okx_instId(sym)
        notional = await _safe(dc.okx_open_interest_usd(inst), None, f"openInterestUSD {inst}")
        oi_usd[sym] = notional

    # market-cap layout
    total_mcap = float(((cg or {}).get("data", {}).get(
        "total_market_cap", {}).get("usd", 0.0)))
    btc_pct = float(((cg or {}).get("data", {}).get(
        "market_cap_percentage", {}).get("btc", 0.0)))
    eth_pct = float(((cg or {}).get("data", {}).get(
        "market_cap_percentage", {}).get("eth", 0.0)))
    btc_mcap = total_mcap * (btc_pct / 100.0) if total_mcap else 0.0
    eth_mcap = total_mcap * (eth_pct / 100.0) if total_mcap else 0.0
    altcap = max(total_mcap - btc_mcap - eth_mcap, 0.0)
    altcap_btc_ratio = (altcap / btc_mcap) if btc_mcap > 0 else 0.0

    # Google Trends + optional on-chain
    trends_avg, trends_by_kw = await _safe(
        dc.google_trends_score(["crypto", "bitcoin", "ethereum"]),
        (0.0, {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0}),
        "google_trends_score"
    )
    mvrv_z = await _safe(dc.glassnode_mvrv_z("BTC"), None, "glassnode_mvrv_z")
    exch_inflow = await _safe(dc.glassnode_exchange_inflow("BTC"), None, "glassnode_exchange_inflow")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_dominance_pct": btc_pct,
        "eth_btc": ethbtc,
        "altcap_btc_ratio": altcap_btc_ratio,
        "funding": funding,
        "open_interest_usd": oi_usd,
        "google_trends_avg7d": float(trends_avg or 0.0),
        "google_trends_breakdown": trends_by_kw or {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0},
        "mvrv_z": mvrv_z,
        "exchange_inflow_proxy": exch_inflow,
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
        fr_pct = float(d.get("lastFundingRate", 0.0)) * 100.0
        lines.append(f"   - {sym}: *{fr_pct:.3f}%*")
    lines.append("• Open Interest (USD est):")
    for sym, notional in m["open_interest_usd"].items():
        lines.append(
            f"   - {sym}: *${notional:,.0f}*" if notional is not None else f"   - {sym}: n/a")
    lines.append(
        f"• Google Trends 7d avg: *{m['google_trends_avg7d']:.1f}* "
        f"(crypto {m['google_trends_breakdown']['crypto']:.1f}, "
        f"bitcoin {m['google_trends_breakdown']['bitcoin']:.1f}, "
        f"ethereum {m['google_trends_breakdown']['ethereum']:.1f})"
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
    # funding (max abs)
    for sym, d in m["funding"].items():
        fr_pct = abs(float(d.get("lastFundingRate", 0.0)) * 100.0)
        if fr_pct >= t["funding_rate_abs_max"]:
            flags.append(f"Funding extreme: {sym} ({fr_pct:.3f}%)")
            break
    # open interest notional USD (if available)
    oi = m["open_interest_usd"]
    if oi.get("BTCUSDT") and oi["BTCUSDT"] >= t["open_interest_btc_usdt_usd_max"]:
        flags.append(f"BTC OI high (${oi['BTCUSDT']:,.0f})")
    if oi.get("ETHUSDT") and oi["ETHUSDT"] >= t["open_interest_eth_usdt_usd_max"]:
        flags.append(f"ETH OI high (${oi['ETHUSDT']:,.0f})")
    # google trends
    if m["google_trends_avg7d"] >= t["google_trends_7d_avg_min"]:
        flags.append("Retail interest (Google) high")
    # on-chain optional
    if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None and m["mvrv_z"] >= t["mvrv_z_extreme"]:
        flags.append("MVRV Z-Score extreme")
    return len(flags), flags

# ───────────────────────────
# Telegram bot commands
# ───────────────────────────


SUBSCRIBERS: set[int] = set()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text(
        "👋 Crypto Cycle Watch online.\n"
        "Commands: /status, /assess, /assess_json, /subscribe, /unsubscribe, "
        "/risk <conservative|moderate|aggressive>, /getrisk, /settime HH:MM."
    )


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
    await update.message.reply_text("✅ Subscribed. You'll receive alerts and daily summaries here.")


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


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show a full, per-metric assessment vs thresholds for this chat's profile."""
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    dc = DataClient()
    try:
        try:
            m = await gather_metrics(dc)
        except Exception as e:
            await update.message.reply_text(
                f"⚠️ Could not fetch metrics right now: {e}\nTry again in a minute."
            )
            return

        t = get_thresholds_for_chat(chat_id)
        profile = (CHAT_PROFILE.get(chat_id, CFG.default_profile)
                   if CFG.risk_profiles else "static")

        def flag_mark(is_flag: bool) -> str:
            return "🚨" if is_flag else "✅"

        lines = []
        lines.append(
            f"🧪 *Current Assessment* (`{m.get('timestamp','n/a')}` UTC)")
        lines.append(f"• Profile: *{profile}*")
        lines.append("")

        dom = float(m.get("btc_dominance_pct") or 0.0)
        f_dom = dom >= float(t["btc_dominance_max"])
        lines.append(
            f"{flag_mark(f_dom)} BTC dominance: *{dom:.2f}%*  (flag ≥ {t['btc_dominance_max']:.2f}%)")

        ethbtc = float(m.get("eth_btc") or 0.0)
        f_ethbtc = ethbtc >= float(t["eth_btc_ratio_max"])
        lines.append(
            f"{flag_mark(f_ethbtc)} ETH/BTC: *{ethbtc:.5f}*  (flag ≥ {t['eth_btc_ratio_max']:.5f})")

        altbtc = float(m.get("altcap_btc_ratio") or 0.0)
        f_altbtc = altbtc >= float(t["altcap_btc_ratio_max"])
        lines.append(
            f"{flag_mark(f_altbtc)} Altcap/BTC: *{altbtc:.2f}*  (flag ≥ {t['altcap_btc_ratio_max']:.2f})")

        funding = m.get("funding") or {}
        max_fr = None
        max_sym = None
        for sym, d in funding.items():
            fr_pct = abs(float(d.get("lastFundingRate") or 0.0) * 100.0)
            if max_fr is None or fr_pct > max_fr:
                max_fr, max_sym = fr_pct, sym
        f_fund = (max_fr or 0.0) >= float(t["funding_rate_abs_max"])
        if max_sym:
            lines.append(
                f"{flag_mark(f_fund)} Funding (max |8h|): *{max_fr:.3f}%* on *{max_sym}*  (flag ≥ {t['funding_rate_abs_max']:.3f}%)")
        else:
            lines.append("✅ Funding: n/a")

        oi = m.get("open_interest_usd") or {}
        btc_oi = oi.get("BTCUSDT")
        eth_oi = oi.get("ETHUSDT")
        f_btc_oi = bool(btc_oi and btc_oi >= float(
            t["open_interest_btc_usdt_usd_max"]))
        f_eth_oi = bool(eth_oi and eth_oi >= float(
            t["open_interest_eth_usdt_usd_max"]))
        lines.append(
            f"{flag_mark(f_btc_oi)} BTC OI est: *${(btc_oi or 0):,.0f}*  (flag ≥ ${t['open_interest_btc_usdt_usd_max']:,.0f})")
        lines.append(
            f"{flag_mark(f_eth_oi)} ETH OI est: *${(eth_oi or 0):,.0f}*  (flag ≥ ${t['open_interest_eth_usdt_usd_max']:,.0f})")

        gavg = float(m.get("google_trends_avg7d") or 0.0)
        f_trends = gavg >= float(t["google_trends_7d_avg_min"])
        lines.append(
            f"{flag_mark(f_trends)} Google Trends 7d avg: *{gavg:.1f}*  (flag ≥ {t['google_trends_7d_avg_min']:.1f})")

        if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None:
            mz = float(m.get("mvrv_z") or 0.0)
            f_mz = mz >= float(t["mvrv_z_extreme"])
            lines.append(
                f"{flag_mark(f_mz)} BTC MVRV Z-Score: *{mz:.2f}*  (flag ≥ {t['mvrv_z_extreme']:.2f})")

        try:
            nflags, flags = evaluate_flags(m, t)
        except Exception:
            nflags, flags = 0, []
        lines.append("")
        if nflags > 0:
            lines.append(
                f"⚠️ *Triggered flags ({nflags}):* " + ", ".join(flags))
        else:
            lines.append("✅ *No flags* at current thresholds.")

        await update.message.reply_markdown("\n".join(lines))
    finally:
        await dc.close()


async def cmd_assess_json(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return a full JSON dump of the current metrics, thresholds, and flags."""
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        t = get_thresholds_for_chat(chat_id)
        nflags, flags = evaluate_flags(m, t)
        profile = (CHAT_PROFILE.get(chat_id, CFG.default_profile)
                   if CFG.risk_profiles else "static")

        payload = {
            "timestamp": m["timestamp"],
            "chat_id": chat_id,
            "profile": profile,
            "thresholds": t,
            "metrics": m,
            "flags_count": nflags,
            "flags": flags,
        }

        # args: "file", "compact", "pretty"
        as_file = False
        pretty = True
        if context.args:
            for a in context.args:
                a = a.lower()
                if a == "file":
                    as_file = True
                elif a == "compact":
                    pretty = False
                elif a == "pretty":
                    pretty = True

        import json
        import io
        text = json.dumps(payload, indent=(
            2 if pretty else None), ensure_ascii=False)

        # Optional webhook
        webhook_status = ""
        webhook = os.getenv("JSON_WEBHOOK_URL")
        if webhook:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.post(webhook, json=payload)
                webhook_status = f" (webhook {r.status_code})"
            except Exception as e:
                webhook_status = f" (webhook error: {e})"

        if as_file or len(text) > 3500:
            bio = io.BytesIO(text.encode("utf-8"))
            fname = f"assess_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            bio.name = fname
            bio.seek(0)
            await update.message.reply_document(document=bio, caption=f"JSON dump{webhook_status}")
        else:
            await update.message.reply_text(text + (f"\n\n{webhook_status}" if webhook_status else ""))
    finally:
        await dc.close()

# ───────────────────────────
# Health server (Koyeb web)
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


def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)

# ───────────────────────────
# Alerts / Summaries
# ───────────────────────────


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

# ───────────────────────────
# Main
# ───────────────────────────


async def main():
    token = os.getenv("TELEGRAM_TOKEN") or CFG.telegram.get("token", "")
    if not token:
        log.error(
            "No TELEGRAM_TOKEN set and no token in config.yml. Set TELEGRAM_TOKEN env.")
        raise SystemExit(1)

    app: Application = ApplicationBuilder().token(token).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("assess", cmd_assess))
    app.add_handler(CommandHandler("assess_json", cmd_assess_json))
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

    # Start PTB and begin polling for updates
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        # Stop polling first, then stop the application
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
