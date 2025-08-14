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
        "funding_rate_abs_max": 0.10,  # percent per 8h
        "open_interest_btc_usdt_usd_max": 20_000_000_000,
        "open_interest_eth_usdt_usd_max": 8_000_000_000,
        "google_trends_7d_avg_min": 75,
        "mvrv_z_extreme": 7.0,
        "fear_greed_greed_min": 70,        # Greed threshold (0-100)
        # 111DMA / (2*350DMA) ratio at/above which we flag
        "pi_cycle_ratio_min": 0.98,
    },
    "glassnode": {"api_key": "", "enable": False},
    "force_chat_id": "",
    "logging": {"level": "INFO"},
    "default_profile": "moderate",
    "risk_profiles": {
        "conservative": {"thresholds": {
            "min_flags_for_alert": 2,
            "btc_dominance_max": 57.0,
            "eth_btc_ratio_max": 0.085,
            "altcap_btc_ratio_max": 1.6,
            "funding_rate_abs_max": 0.06,
            "open_interest_btc_usdt_usd_max": 15_000_000_000,
            "open_interest_eth_usdt_usd_max": 6_000_000_000,
            "google_trends_7d_avg_min": 60,
            "mvrv_z_extreme": 6.5,
            "fear_greed_greed_min": 60,
            "pi_cycle_ratio_min": 0.96,
        }},
        "moderate": {"thresholds": {
            "min_flags_for_alert": 3,
            "btc_dominance_max": 60.0,
            "eth_btc_ratio_max": 0.09,
            "altcap_btc_ratio_max": 1.8,
            "funding_rate_abs_max": 0.10,
            "open_interest_btc_usdt_usd_max": 20_000_000_000,
            "open_interest_eth_usdt_usd_max": 8_000_000_000,
            "google_trends_7d_avg_min": 75,
            "mvrv_z_extreme": 7.0,
            "fear_greed_greed_min": 70,
            "pi_cycle_ratio_min": 0.98,
        }},
        "aggressive": {"thresholds": {
            "min_flags_for_alert": 4,
            "btc_dominance_max": 62.0,
            "eth_btc_ratio_max": 0.095,
            "altcap_btc_ratio_max": 2.0,
            "funding_rate_abs_max": 0.14,
            "open_interest_btc_usdt_usd_max": 25_000_000_000,
            "open_interest_eth_usdt_usd_max": 10_000_000_000,
            "google_trends_7d_avg_min": 85,
            "mvrv_z_extreme": 8.0,
            "fear_greed_greed_min": 80,
            "pi_cycle_ratio_min": 1.00,
        }},
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
HEADERS = {"User-Agent": "crypto-cycle-bot/1.3", "Accept": "application/json"}


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

    # BTC daily closes for Pi Cycle (CoinGecko)
    async def coingecko_btc_daily(self, days: int = 500) -> List[float]:
        data = await fetch_json(
            self.client,
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            {"vs_currency": "usd", "days": days, "interval": "daily"},
        )
        prices = data.get("prices", [])  # list[[ms, price], ...]
        return [float(p[1]) for p in prices]

    # OKX v5 public endpoints (no API key needed)
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
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/funding-rate", {"instId": instId})
        try:
            lst = data.get("data", [])
            if lst:
                # decimal; 0.001 == 0.1% per 8h
                return float(lst[0]["fundingRate"])
        except Exception:
            pass
        return None

    async def okx_open_interest_usd(self, instId: str) -> Optional[float]:
        # include instType; prefer oiUsd; fallback to oiCcy * last price
        data = await fetch_json(
            self.client,
            f"{OKX_BASE}/api/v5/public/open-interest",
            {"instType": "SWAP", "instId": instId}
        )
        try:
            lst = data.get("data", [])
            if not lst:
                return None
            row = lst[0]
            val = row.get("oiUsd")
            if val not in (None, "", "0"):
                return float(val)
            oi_ccy = row.get("oiCcy")
            if oi_ccy not in (None, ""):
                last = await _safe(self.okx_ticker_last_price(instId), None, f"lastPrice {instId}")
                if last:
                    return float(oi_ccy) * float(last)
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

    # Fear & Greed (alternative.me)
    async def fear_greed(self) -> Tuple[Optional[int], Optional[str]]:
        data = await fetch_json(self.client, "https://api.alternative.me/fng/", {"limit": 1})
        try:
            row = data.get("data", [])[0]
            return int(row["value"]), str(row["value_classification"])
        except Exception:
            return None, None

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


def _sma(vals: List[float], window: int) -> Optional[float]:
    if len(vals) < window:
        return None
    return sum(vals[-window:]) / float(window)


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

    # Google Trends + Fear & Greed + optional on-chain
    trends_avg, trends_by_kw = await _safe(
        dc.google_trends_score(["crypto", "bitcoin", "ethereum"]),
        (0.0, {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0}),
        "google_trends_score"
    )
    fng_value, fng_label = await _safe(dc.fear_greed(), (None, None), "fear_greed")
    mvrv_z = await _safe(dc.glassnode_mvrv_z("BTC"), None, "glassnode_mvrv_z")
    exch_inflow = await _safe(dc.glassnode_exchange_inflow("BTC"), None, "glassnode_exchange_inflow")

    # Pi Cycle Top proximity (BTC 111DMA vs 2×350DMA)
    btc_daily = await _safe(dc.coingecko_btc_daily(500), [], "btc_daily_prices")
    sma111 = _sma(btc_daily, 111)
    sma350 = _sma(btc_daily, 350)
    pi_ratio = None
    two_350 = None
    if sma111 is not None and sma350 is not None and sma350 > 0:
        two_350 = 2.0 * sma350
        pi_ratio = sma111 / two_350

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "btc_dominance_pct": btc_pct,
        "eth_btc": ethbtc,
        "altcap_btc_ratio": altcap_btc_ratio,
        "funding": funding,
        "open_interest_usd": oi_usd,
        "google_trends_avg7d": float(trends_avg or 0.0),
        "google_trends_breakdown": trends_by_kw or {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0},
        "fear_greed_index": fng_value,
        "fear_greed_label": fng_label,
        "mvrv_z": mvrv_z,
        "exchange_inflow_proxy": exch_inflow,
        "pi_cycle": {
            "sma111": sma111,
            "sma350x2": two_350,
            # == 111DMA / (2*350DMA); >=1.00 is a cross/trigger
            "ratio": pi_ratio,
        },
    }

# ───────────────────────────
# Presentation helpers (HTML)
# ───────────────────────────


def bullet(flagged: bool) -> str:
    return "🔴" if flagged else "🟢"


def b(txt: str) -> str:
    return f"<b>{txt}</b>"


def fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def build_status_text(m: Dict[str, Any], t: Dict[str, Any] | None = None) -> str:
    """Human-friendly snapshot with colored bullets on metric values."""
    t = t or CFG.thresholds

    # Flags (for coloring)
    flags_count, flags_list = evaluate_flags(m, t)

    # Per-metric local flag checks
    dom = m["btc_dominance_pct"]
    f_dom = dom >= t["btc_dominance_max"]
    ethbtc = m["eth_btc"]
    f_ethbtc = ethbtc >= t["eth_btc_ratio_max"]
    altbtc = m["altcap_btc_ratio"]
    f_altbtc = altbtc >= t["altcap_btc_ratio_max"]

    # funding (max abs)
    funding = m["funding"]
    max_fr = 0.0
    max_sym = None
    for sym, d in funding.items():
        fr_pct = abs(float(d.get("lastFundingRate", 0.0)) * 100.0)
        if fr_pct > max_fr:
            max_fr, max_sym = fr_pct, sym
    f_fund = max_fr >= t["funding_rate_abs_max"]

    # open interest
    oi = m["open_interest_usd"]
    btc_oi = oi.get("BTCUSDT")
    eth_oi = oi.get("ETHUSDT")
    f_btc_oi = bool(btc_oi and btc_oi >= t["open_interest_btc_usdt_usd_max"])
    f_eth_oi = bool(eth_oi and eth_oi >= t["open_interest_eth_usdt_usd_max"])

    # Google Trends
    gavg = m["google_trends_avg7d"]
    f_trends = gavg >= t["google_trends_7d_avg_min"]

    # Fear & Greed
    fng_val = m.get("fear_greed_index")
    fng_lab = m.get("fear_greed_label")
    f_fng = bool(fng_val is not None and fng_val >=
                 t.get("fear_greed_greed_min", 70))

    # Pi Cycle
    pi = m.get("pi_cycle") or {}
    pi_ratio = pi.get("ratio")
    pi_flag = bool(pi_ratio is not None and pi_ratio >=
                   t.get("pi_cycle_ratio_min", 0.98))

    lines: List[str] = []
    lines.append(f"📊 {b('Crypto Market Snapshot')} ({m['timestamp']} UTC)")

    lines.append(
        f"• Bitcoin market share of total crypto market: "
        f"{bullet(f_dom)} {b(f'{dom:.2f}%')}  (flag ≥ {t['btc_dominance_max']:.2f}%)"
    )

    lines.append(
        f"• Ether price relative to Bitcoin (ETH/BTC): "
        f"{bullet(f_ethbtc)} {b(f'{ethbtc:.5f}')}  (flag ≥ {t['eth_btc_ratio_max']:.5f})"
    )

    lines.append(
        f"• Altcoin market cap / Bitcoin market cap: "
        f"{bullet(f_altbtc)} {b(f'{altbtc:.2f}')}  (flag ≥ {t['altcap_btc_ratio_max']:.2f})"
    )

    if max_sym:
        lines.append(
            f"• Perpetual funding rate (max absolute across watched pairs): "
            f"{bullet(f_fund)} {b(f'{max_fr:.3f}%')} on {max_sym}  "
            f"(flag ≥ {t['funding_rate_abs_max']:.3f}%)"
        )
    else:
        lines.append(f"• Perpetual funding rate: {bullet(False)} {b('n/a')}")

    lines.append(
        f"• Bitcoin open interest (USD, estimate): "
        f"{bullet(f_btc_oi)} {b(fmt_usd(btc_oi))}  "
        f"(flag ≥ {fmt_usd(t['open_interest_btc_usdt_usd_max'])})"
    )

    lines.append(
        f"• Ether open interest (USD, estimate): "
        f"{bullet(f_eth_oi)} {b(fmt_usd(eth_oi))}  "
        f"(flag ≥ {fmt_usd(t['open_interest_eth_usdt_usd_max'])})"
    )

    lines.append(
        f"• Google Trends (7-day average; crypto/bitcoin/ethereum): "
        f"{bullet(f_trends)} {b(f'{gavg:.1f}')}  "
        f"(flag ≥ {t['google_trends_7d_avg_min']:.1f})"
    )

    if fng_val is not None:
        lines.append(
            f"• Fear & Greed Index (overall crypto): "
            f"{bullet(f_fng)} {b(str(fng_val))} ({fng_lab or 'n/a'})  "
            f"(flag ≥ {t.get('fear_greed_greed_min', 70)})"
        )

    # Pi Cycle line
    if pi_ratio is not None and pi.get("sma111") and pi.get("sma350x2"):
        pct = pi_ratio * 100.0
        th_pct = t.get("pi_cycle_ratio_min", 0.98) * 100.0
        lines.append(
            f"• Pi Cycle Top proximity (111-day MA vs 2×350-day MA): "
            f"{bullet(pi_flag)} {b(f'{pct:.1f}% of cross')} "
            f"(111DMA {fmt_usd(pi['sma111'])}, 2×350DMA {fmt_usd(pi['sma350x2'])}; flag ≥ {th_pct:.1f}%)"
        )
    else:
        lines.append("• Pi Cycle Top proximity: 🟢 " + b("n/a"))

    if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None:
        mz_val = m["mvrv_z"]
        f_mz = mz_val >= t["mvrv_z_extreme"]
        lines.append(
            f"• Bitcoin MVRV Z-Score (on-chain valuation): "
            f"{bullet(f_mz)} {b(f'{mz_val:.2f}')}  "
            f"(flag ≥ {t['mvrv_z_extreme']:.2f})"
        )

    if flags_count > 0:
        lines.append("")
        lines.append(
            f"⚠️ {b(f'Triggered flags ({flags_count})')}: " + ", ".join(flags_list))

    return "\n".join(lines)


def evaluate_flags(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[str]]:
    flags: List[str] = []
    if m["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("High Bitcoin dominance")
    if m["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("Elevated ETH/BTC ratio")
    if m["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Alt market cap stretched vs BTC")
    # funding (max abs)
    for sym, d in m["funding"].items():
        fr_pct = abs(float(d.get("lastFundingRate", 0.0)) * 100.0)
        if fr_pct >= t["funding_rate_abs_max"]:
            flags.append(f"Perpetual funding extreme on {sym} ({fr_pct:.3f}%)")
            break
    # open interest notional USD (if available)
    oi = m["open_interest_usd"]
    if oi.get("BTCUSDT") and oi["BTCUSDT"] >= t["open_interest_btc_usdt_usd_max"]:
        flags.append(f"High Bitcoin open interest (${oi['BTCUSDT']:,.0f})")
    if oi.get("ETHUSDT") and oi["ETHUSDT"] >= t["open_interest_eth_usdt_usd_max"]:
        flags.append(f"High Ether open interest (${oi['ETHUSDT']:,.0f})")
    # google trends
    if m["google_trends_avg7d"] >= t["google_trends_7d_avg_min"]:
        flags.append("Elevated retail interest (Google Trends)")
    # fear & greed
    fg = m.get("fear_greed_index")
    if fg is not None and fg >= t.get("fear_greed_greed_min", 70):
        flags.append("Greed is elevated (Fear & Greed Index)")
    # Pi Cycle
    pi = m.get("pi_cycle") or {}
    if pi.get("ratio") is not None and pi["ratio"] >= t.get("pi_cycle_ratio_min", 0.98):
        flags.append("Pi Cycle Top proximity elevated")
    # on-chain optional
    if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None and m["mvrv_z"] >= t["mvrv_z_extreme"]:
        flags.append("On-chain overvaluation (MVRV Z-Score)")
    return len(flags), flags

# ───────────────────────────
# Telegram bot commands
# ───────────────────────────


SUBSCRIBERS: set[int] = set()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    msg = (
        "👋 Crypto Cycle Watch online.\n"
        "Commands: /status, /assess, /assess_json, /subscribe, /unsubscribe, "
        "/risk &lt;conservative|moderate|aggressive&gt;, /getrisk, /settime HH:MM."
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        t = get_thresholds_for_chat(update.effective_chat.id)
        text = build_status_text(m, t)
        if CFG.risk_profiles:
            cur = CHAT_PROFILE.get(
                update.effective_chat.id, CFG.default_profile)
            text += f"\n• Active risk profile: {b(cur)}"
        await update.message.reply_text(text, parse_mode="HTML")
    finally:
        await dc.close()


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed. You'll receive alerts and daily summaries here.", parse_mode="HTML")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.discard(update.effective_chat.id)
    await update.message.reply_text("🛑 Unsubscribed.", parse_mode="HTML")


async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args[0]) != 5 or context.args[0][2] != ":":
        await update.message.reply_text("Usage: /settime HH:MM (24h)", parse_mode="HTML")
        return
    CFG.schedule["daily_summary_time"] = context.args[0]
    await update.message.reply_text(f"🕒 Daily summary time set to {b(context.args[0])} (server time).", parse_mode="HTML")


async def cmd_risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not CFG.risk_profiles:
        await update.message.reply_text("Risk profiles not configured; using static thresholds.", parse_mode="HTML")
        return
    if not context.args:
        await update.message.reply_text("Usage: /risk &lt;conservative|moderate|aggressive&gt;", parse_mode="HTML")
        return
    choice = context.args[0].lower()
    if choice not in CFG.risk_profiles:
        await update.message.reply_text(f"Unknown profile '{choice}'. Options: {', '.join(CFG.risk_profiles.keys())}", parse_mode="HTML")
        return
    CHAT_PROFILE[update.effective_chat.id] = choice
    save_profiles()
    await update.message.reply_text(f"✅ Risk profile set to {b(choice)}.", parse_mode="HTML")


async def cmd_getrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cur = CHAT_PROFILE.get(update.effective_chat.id, CFG.default_profile)
    await update.message.reply_text(f"Current risk profile: {b(cur)}", parse_mode="HTML")


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full, per-metric assessment vs thresholds for this chat's profile."""
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    dc = DataClient()
    try:
        try:
            m = await gather_metrics(dc)
        except Exception as e:
            await update.message.reply_text(
                f"⚠️ Could not fetch metrics right now: {e}\nTry again in a minute.",
                parse_mode="HTML"
            )
            return

        t = get_thresholds_for_chat(chat_id)
        profile = (CHAT_PROFILE.get(chat_id, CFG.default_profile)
                   if CFG.risk_profiles else "static")

        text = build_status_text(m, t)
        nflags, flags = evaluate_flags(m, t)
        lines = [text, "", f"• Active risk profile: {b(profile)}"]
        if nflags > 0:
            lines.append(
                f"⚠️ {b(f'Triggered flags ({nflags})')}: " + ", ".join(flags))
        else:
            lines.append("✅ No flags at current thresholds.")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
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
            # No parse_mode for JSON text to avoid HTML escaping issues
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
        t = get_thresholds_for_chat(None)
        text = build_status_text(m, t)
        n, flags = evaluate_flags(m, t)
        if n > 0:
            text += "\n\n" + \
                f"⚠️ {b(f'Triggered flags ({n})')}: " + ", ".join(flags)
        for chat_id in set(SUBSCRIBERS):
            try:
                await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
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
            text = "🚨 " + b("Top Risk Alert") + "\n" + build_status_text(m, t) + "\n\n" + \
                   f"⚠️ {b(f'Triggered flags ({n})')}: " + ", ".join(flags)
            for chat_id in set(SUBSCRIBERS):
                try:
                    await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
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
            await app.bot.send_message(chat_id=int(CFG.force_chat_id), text="🤖 Crypto Cycle Watch bot started.", parse_mode="HTML")
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
