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
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config / Models
# ─────────────────────────────────────────────────────────────────────────────


class Config(BaseModel):
    telegram: Dict[str, Any]
    schedule: Dict[str, Any]
    symbols: Dict[str, List[str]]
    thresholds: Dict[str, Any]  # back-compat if risk_profiles not used
    glassnode: Dict[str, Any] = {}
    force_chat_id: str = ""
    logging: Dict[str, Any] = {"level": "INFO"}
    default_profile: str = "moderate"
    risk_profiles: Dict[str, Dict[str, Any]] | None = None


def load_config(path: str = "config.yml") -> "Config":
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


CFG = load_config()

# Per-chat risk profile storage
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
        profile = CFG.default_profile
        if chat_id is not None and chat_id in CHAT_PROFILE:
            profile = CHAT_PROFILE[chat_id]
        prof = CFG.risk_profiles.get(
            profile, CFG.risk_profiles.get(CFG.default_profile))
        if prof and isinstance(prof, dict):
            return prof.get("thresholds", CFG.thresholds)
    return CFG.thresholds


# Logging
logging.basicConfig(level=getattr(logging, CFG.logging.get("level", "INFO")))
log = logging.getLogger("crypto-cycle-bot")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP utils
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# Data Providers
# ─────────────────────────────────────────────────────────────────────────────


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient()

    async def close(self):
        await self.client.aclose()

    async def coingecko_global(self) -> Dict[str, Any]:
        url = "https://api.coingecko.com/api/v3/global"
        return await fetch_json(self.client, url)

    async def binance_price(self, symbol: str) -> float:
        url = "https://api.binance.com/api/v3/ticker/price"
        data = await fetch_json(self.client, url, params={"symbol": symbol})
        return float(data["price"])

    async def binance_premium_index(self, symbol: str) -> Dict[str, Any]:
        url = "https://fapi.binance.com/fapi/v1/premiumIndex"
        return await fetch_json(self.client, url, params={"symbol": symbol})

    async def binance_open_interest(self, symbol: str) -> Dict[str, Any]:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        return await fetch_json(self.client, url, params={"symbol": symbol})

    async def google_trends_score(self, keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        from pytrends.request import TrendReq
        import pandas as pd

        def _work():
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload(
                keywords, cat=0, timeframe="now 7-d", geo="", gprop="")
            df = pytrends.interest_over_time()
            if df.empty:
                return 0.0, {k: 0.0 for k in keywords}
            out = {k: float(df[k].mean()) for k in keywords}
            avg = float(pd.Series(out).mean())
            return avg, out

        return await asyncio.to_thread(_work)

    async def glassnode_mvrv_z(self, asset: str = "BTC") -> Optional[float]:
        api_key = CFG.glassnode.get("api_key") or ""
        if not api_key or not CFG.glassnode.get("enable"):
            return None
        url = "https://api.glassnode.com/v1/metrics/market/mvrv_z_score"
        data = await fetch_json(self.client, url, params={"a": asset, "api_key": api_key, "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

    async def glassnode_exchange_inflow(self, asset: str = "BTC") -> Optional[float]:
        api_key = CFG.glassnode.get("api_key") or ""
        if not api_key or not CFG.glassnode.get("enable"):
            return None
        url = "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_to_exchanges_adjusted_sum"
        data = await fetch_json(self.client, url, params={"a": asset, "api_key": api_key, "i": "24h"})
        if isinstance(data, list) and data:
            return float(data[-1]["v"])
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────


async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    tasks = []
    tasks.append(dc.coingecko_global())
    tasks.append(dc.binance_price("ETHBTC"))
    for sym in CFG.symbols["funding"]:
        tasks.append(dc.binance_premium_index(sym))
    for sym in CFG.symbols["oi"]:
        tasks.append(dc.binance_open_interest(sym))
    tasks.append(dc.google_trends_score(["crypto", "bitcoin", "ethereum"]))
    tasks.append(dc.glassnode_mvrv_z("BTC"))
    tasks.append(dc.glassnode_exchange_inflow("BTC"))

    results = await asyncio.gather(*tasks)

    idx = 0
    cg = results[idx]
    idx += 1
    ethbtc = results[idx]
    idx += 1

    funding = {}
    for sym in CFG.symbols["funding"]:
        funding[sym] = results[idx]
        idx += 1

    oi = {}
    for sym in CFG.symbols["oi"]:
        oi[sym] = results[idx]
        idx += 1

    trends_avg, trends_by_kw = results[idx]
    idx += 1
    mvrv_z = results[idx]
    idx += 1
    exch_inflow = results[idx]
    idx += 1

    total_mcap = cg["data"]["total_market_cap"].get("usd", 0.0)
    btc_pct = cg["data"]["market_cap_percentage"].get("btc", 0.0)
    eth_pct = cg["data"]["market_cap_percentage"].get("eth", 0.0)
    btc_mcap = total_mcap * (btc_pct / 100.0)
    eth_mcap = total_mcap * (eth_pct / 100.0)
    altcap = max(total_mcap - btc_mcap - eth_mcap, 0.0)
    altcap_btc_ratio = (altcap / btc_mcap) if btc_mcap > 0 else 0.0

    def _oi_usd(symbol: str) -> Optional[float]:
        try:
            raw = oi[symbol]
            qty = float(raw.get("openInterest", 0.0))
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

    lines.append("• Funding (8h, est):")
    for sym, d in metrics["funding"].items():
        fr = float(d.get("lastFundingRate", 0.0)) * 100
        lines.append(f"   - {sym}: *{fr:.3f}%*")

    lines.append("• Open Interest (USD est):")
    for sym, notional in metrics["open_interest_usd"].items():
        if notional is not None:
            lines.append(f"   - {sym}: *${notional:,.0f}*")
        else:
            lines.append(f"   - {sym}: n/a")

    lines.append(
        f"• Google Trends 7d avg: *{metrics['google_trends_avg7d']:.1f}* "
        f"(crypto {metrics['google_trends_breakdown']['crypto']:.1f}, "
        f"bitcoin {metrics['google_trends_breakdown']['bitcoin']:.1f}, "
        f"ethereum {metrics['google_trends_breakdown']['ethereum']:.1f})"
    )

    if metrics.get("mvrv_z") is not None:
        lines.append(
            f"• BTC MVRV Z-Score (Glassnode): *{metrics['mvrv_z']:.2f}*")
    if metrics.get("exchange_inflow_proxy") is not None:
        lines.append(
            f"• Exchange inflow proxy (Glassnode, adj sum): *{metrics['exchange_inflow_proxy']:.2f}*")

    return "\n".join(lines)


def evaluate_flags(metrics: Dict[str, Any], thresholds: Dict[str, Any]) -> Tuple[int, List[str]]:
    t = thresholds
    flags: List[str] = []

    if metrics["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("BTC dominance high")

    if metrics["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("ETH/BTC elevated")

    if metrics["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Altcap/BTC stretched")

    for sym, d in metrics["funding"].items():
        fr = abs(float(d.get("lastFundingRate", 0.0)) * 100)
        if fr >= t["funding_rate_abs_max"]:
            flags.append(f"Funding extreme: {sym} ({fr:.3f}%)")
            break

    oi_usd = metrics["open_interest_usd"]
    btc_oi = oi_usd.get("BTCUSDT")
    eth_oi = _
