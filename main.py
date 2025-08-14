import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

# =========================
# Defaults & Configuration
# =========================

DEFAULTS: Dict[str, Any] = {
    "telegram": {"token": ""},
    "schedule": {"daily_summary_time": "13:00"},  # UTC
    "symbols": {
        "funding": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
        "oi": ["BTCUSDT", "ETHUSDT"],
        # Equal-weight ALT basket (spot pairs on OKX)
        "alt_basket": [
            "SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
            "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"
        ],
    },
    "thresholds": {
        # general 3-color banding
        "warn_fraction": 0.80,
        "min_flags_for_alert": 3,

        # Market structure (risk when high)
        "btc_dominance_max": 60.0,
        "eth_btc_ratio_max": 0.09,
        "altcap_btc_ratio_max": 1.8,

        # Derivatives (risk when high)
        "funding_rate_abs_max": 0.10,  # percent per 8h
        "open_interest_btc_usdt_usd_max": 20_000_000_000,
        "open_interest_eth_usdt_usd_max": 8_000_000_000,

        # Sentiment (risk when high)
        "google_trends_7d_avg_min": 75,
        "fear_greed_greed_min": 70,
        "fear_greed_ma14_min": 70,
        "fear_greed_ma30_min": 65,
        "greed_persistence_days": 10,
        "greed_persistence_pct30_min": 0.60,

        # Cycle/On-chain (risk when high)
        "mvrv_z_extreme": 7.0,
        "pi_cycle_ratio_min": 0.98,  # 111DMA / (2*350DMA)

        # Momentum & Fibonacci (weekly)
        "btc_weekly_rsi_flag": 70.0,
        "btc_weekly_rsi_warn": 60.0,
        "ethbtc_weekly_rsi_flag": 65.0,
        "ethbtc_weekly_rsi_warn": 55.0,
        "alt_weekly_rsi_flag": 75.0,   # ALT basket thresholds
        "alt_weekly_rsi_warn": 65.0,
        "stochrsi_overbought": 0.80,   # K/D zone
        "fib_warn_pct": 0.03,          # 3% away (warn if nearer)
        "fib_flag_pct": 0.015,         # 1.5% away (flag if nearer)

        # Composite score thresholds
        "composite_warn": 40,          # Yellow at/above this score
        "composite_flag": 70           # Red at/above this score
    },
    "glassnode": {"api_key": "", "enable": False},
    "force_chat_id": "",
    "logging": {"level": "INFO"},
    "default_profile": "moderate",
    "risk_profiles": {
        "conservative": {"thresholds": {
            "warn_fraction": 0.75,
            "min_flags_for_alert": 2,
            "btc_dominance_max": 57.0,
            "eth_btc_ratio_max": 0.085,
            "altcap_btc_ratio_max": 1.6,
            "funding_rate_abs_max": 0.06,
            "open_interest_btc_usdt_usd_max": 15_000_000_000,
            "open_interest_eth_usdt_usd_max": 6_000_000_000,
            "google_trends_7d_avg_min": 60,
            "fear_greed_greed_min": 60,
            "fear_greed_ma14_min": 65,
            "fear_greed_ma30_min": 60,
            "greed_persistence_days": 8,
            "greed_persistence_pct30_min": 0.50,
            "mvrv_z_extreme": 6.5,
            "pi_cycle_ratio_min": 0.96,
            "btc_weekly_rsi_flag": 68.0,
            "btc_weekly_rsi_warn": 58.0,
            "ethbtc_weekly_rsi_flag": 63.0,
            "ethbtc_weekly_rsi_warn": 53.0,
            "alt_weekly_rsi_flag": 72.0,
            "alt_weekly_rsi_warn": 62.0,
            "fib_warn_pct": 0.035,
            "fib_flag_pct": 0.02,
            "composite_warn": 35,
            "composite_flag": 60
        }},
        "moderate": {"thresholds": {}},  # use defaults above
        "aggressive": {"thresholds": {
            "warn_fraction": 0.85,
            "min_flags_for_alert": 4,
            "btc_dominance_max": 62.0,
            "eth_btc_ratio_max": 0.095,
            "altcap_btc_ratio_max": 2.0,
            "funding_rate_abs_max": 0.14,
            "open_interest_btc_usdt_usd_max": 25_000_000_000,
            "open_interest_eth_usdt_usd_max": 10_000_000_000,
            "google_trends_7d_avg_min": 85,
            "fear_greed_greed_min": 80,
            "fear_greed_ma14_min": 75,
            "fear_greed_ma30_min": 70,
            "greed_persistence_days": 14,
            "greed_persistence_pct30_min": 0.70,
            "mvrv_z_extreme": 8.0,
            "pi_cycle_ratio_min": 1.00,
            "btc_weekly_rsi_flag": 75.0,
            "btc_weekly_rsi_warn": 65.0,
            "ethbtc_weekly_rsi_flag": 70.0,
            "ethbtc_weekly_rsi_warn": 60.0,
            "alt_weekly_rsi_flag": 78.0,
            "alt_weekly_rsi_warn": 68.0,
            "fib_warn_pct": 0.025,
            "fib_flag_pct": 0.012,
            "composite_warn": 50,
            "composite_flag": 80
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


def deep_merge(base, extra):
    if isinstance(base, dict) and isinstance(extra, dict):
        out = dict(base)
        for k, v in extra.items():
            out[k] = deep_merge(base.get(k), v)
        return out
    return extra if extra is not None else base


def load_config(path: str = "config.yml") -> Config:
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
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

# per-chat profile persistence
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


def get_profile_for_chat(chat_id: Optional[int]) -> str:
    return CHAT_PROFILE.get(chat_id, CFG.default_profile) if CFG.risk_profiles else "static"


def get_thresholds_for_chat(chat_id: Optional[int]) -> Dict[str, Any]:
    base = CFG.thresholds
    if CFG.risk_profiles and chat_id in CHAT_PROFILE:
        prof = CFG.risk_profiles.get(CHAT_PROFILE[chat_id], {})
        return deep_merge(base, prof.get("thresholds", {}))
    if CFG.risk_profiles:
        prof = CFG.risk_profiles.get(CFG.default_profile, {})
        return deep_merge(base, prof.get("thresholds", {}))
    return base

# =========================
# HTTP & Data Providers
# =========================


OKX_BASE = "https://www.okx.com"
HEADERS = {"User-Agent": "crypto-cycle-bot/2.0", "Accept": "application/json"}


def to_okx_instId(sym: str) -> str:
    s = sym.upper()
    if s.endswith("USDT"):
        base, quote = s[:-4], "USDT"   # fixed: 4 chars
    elif s.endswith("USD"):
        base, quote = s[:-3], "USD"
    else:
        base, quote = s, "USDT"
    return f"{base}-{quote}-SWAP"


def to_okx_uly(sym: str) -> str:
    s = sym.upper()
    if s.endswith("USDT"):
        base, quote = s[:-4], "USDT"   # fixed: 4 chars
    elif s.endswith("USD"):
        base, quote = s[:-3], "USD"
    else:
        base, quote = s, "USDT"
    return f"{base}-{quote}"


async def fetch_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> Any:
    for attempt in range(3):
        try:
            hh = dict(HEADERS)
            if headers:
                hh.update(headers)
            r = await client.get(url, params=params, headers=hh, timeout=25)
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

    # CoinGecko
    async def coingecko_global(self) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://api.coingecko.com/api/v3/global")

    async def coingecko_eth_btc(self) -> float:
        data = await fetch_json(self.client, "https://api.coingecko.com/api/v3/simple/price",
                                {"ids": "bitcoin,ethereum", "vs_currencies": "usd"})
        eth_usd = float(data["ethereum"]["usd"])
        btc_usd = float(data["bitcoin"]["usd"])
        return eth_usd / btc_usd

    async def coingecko_btc_daily(self, days: str | int = "max") -> List[float]:
        data = await fetch_json(self.client, "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                                {"vs_currency": "usd", "days": days, "interval": "daily"})
        return [float(p[1]) for p in data.get("prices", [])]

    # CoinMarketCap (conditional on key)
    def _cmc_headers(self) -> Dict[str, str]:
        key = os.getenv("CMC_API_KEY", "")
        return {"X-CMC_PRO_API_KEY": key} if key else {}

    async def cmc_global(self) -> Optional[Dict[str, Any]]:
        if not self._cmc_headers():
            return None
        return await fetch_json(self.client, "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                                headers=self._cmc_headers())

    async def cmc_quotes(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        if not self._cmc_headers():
            return None
        syms = ",".join(symbols)
        return await fetch_json(self.client, "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                                params={"symbol": syms, "convert": "USD"},
                                headers=self._cmc_headers())

    # CoinPaprika
    async def paprika_global(self) -> Optional[Dict[str, Any]]:
        return await fetch_json(self.client, "https://api.coinpaprika.com/v1/global")

    # OKX public
    async def okx_ticker_last_price(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/market/ticker", {"instId": instId})
        lst = data.get("data", [])
        if lst:
            return float(lst[0]["last"])
        return None

    async def okx_eth_btc_ratio(self) -> Optional[float]:
        price = await self.okx_ticker_last_price("ETH-BTC")
        return float(price) if price is not None else None

    async def okx_funding_rate(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/funding-rate", {"instId": instId})
        lst = data.get("data", [])
        if lst:
            return float(lst[0]["fundingRate"])
        return None

    async def okx_instruments_map(self, uly: str) -> Dict[str, Any]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/instruments",
                                {"instType": "SWAP", "uly": uly})
        return {row.get("instId"): row for row in data.get("data", [])}

    async def okx_open_interest_usd_by_inst(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/open-interest",
                                {"instType": "SWAP", "instId": instId})
        rows = data.get("data", [])
        if not rows:
            return None
        row = rows[0]
        val = row.get("oiUsd")
        if val not in (None, "", "0"):
            return float(val)
        # fallback: oiCcy * last
        oi_ccy = row.get("oiCcy")
        if oi_ccy not in (None, "", "0"):
            last = await _safe(self.okx_ticker_last_price(instId), None, f"lastPrice {instId}")
            if last:
                return float(oi_ccy) * float(last)
        return None

    async def okx_open_interest_usd_by_uly(self, uly: str) -> Optional[float]:
        oi_data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/public/open-interest",
                                   {"instType": "SWAP", "uly": uly})
        rows = oi_data.get("data", []) or []
        if not rows:
            return None
        inst_map = await self.okx_instruments_map(uly)
        total = 0.0
        for r in rows:
            inst = r.get("instId")
            if not inst:
                continue
            try:
                oi_contracts = float(r.get("oi", 0) or 0)
            except Exception:
                oi_contracts = 0.0
            if oi_contracts <= 0:
                continue
            spec = inst_map.get(inst, {})
            try:
                ct_val = float(spec.get("ctVal", 0) or 0)
            except Exception:
                ct_val = 0.0
            ct_val_ccy = spec.get("ctValCcy") or ""
            notional = 0.0
            if ct_val > 0:
                if ct_val_ccy.upper() in ("USD", "USDT"):
                    notional = oi_contracts * ct_val
                else:
                    last = await _safe(self.okx_ticker_last_price(inst), None, f"lastPrice {inst}")
                    if last:
                        notional = oi_contracts * ct_val * float(last)
            else:
                try:
                    oi_ccy_amt = float(r.get("oiCcy", 0) or 0)
                except Exception:
                    oi_ccy_amt = 0.0
                if oi_ccy_amt > 0:
                    last = await _safe(self.okx_ticker_last_price(inst), None, f"lastPrice {inst}")
                    if last:
                        notional = oi_ccy_amt * float(last)
            total += notional
        return total if total > 0 else None

    async def okx_weekly_closes(self, instId: str, limit: int = 400) -> List[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/market/candles",
                                {"instId": instId, "bar": "1W", "limit": limit})
        rows = data.get("data", [])
        try:
            rows = sorted(rows, key=lambda r: int(r[0]))  # ascending time
        except Exception:
            pass
        return [float(r[4]) for r in rows if len(r) > 4]

    async def okx_daily_closes(self, instId: str, limit: int = 500) -> List[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/market/candles",
                                {"instId": instId, "bar": "1D", "limit": limit})
        rows = data.get("data", [])
        try:
            rows = sorted(rows, key=lambda r: int(r[0]))  # ascending time
        except Exception:
            pass
        return [float(r[4]) for r in rows if len(r) > 4]

    # Google Trends
    async def google_trends_score(self, keywords: List[str]) -> Tuple[float, Dict[str, float]]:
        def _fetch():
            try:
                from pytrends.request import TrendReq  # type: ignore
                import pandas as pd  # type: ignore
                py = TrendReq(hl="en-US", tz=0)
                py.build_payload(kw_list=keywords, timeframe="now 7-d", geo="")
                df = py.interest_over_time()
                if df is None or df.empty:
                    return 0.0, {k: 0.0 for k in keywords}
                means = {k: float(pd.to_numeric(
                    df[k], errors="coerce").mean()) for k in keywords}
                avg = sum(means.values()) / max(len(means), 1)
                return float(avg), {k: float(v) for k, v in means.items()}
            except Exception as e:
                log.warning("pytrends failed: %s", e)
                return 0.0, {k: 0.0 for k in keywords}
        return await asyncio.to_thread(_fetch)

    # Fear & Greed
    async def fear_greed(self) -> Tuple[Optional[int], Optional[str]]:
        try:
            data = await fetch_json(self.client, "https://api.alternative.me/fng/", {"limit": 1})
            row = (data or {}).get("data", [])[0]
            return int(row["value"]), str(row.get("value_classification") or "")
        except Exception as e:
            log.warning("fear_greed fetch failed: %s", e)
            return None, None

    async def fear_greed_history(self, days: int = 60) -> List[int]:
        try:
            data = await fetch_json(self.client, "https://api.alternative.me/fng/", {"limit": days})
            vals = []
            for r in (data or {}).get("data", []):
                try:
                    vals.append(int(r["value"]))
                except Exception:
                    pass
            return vals
        except Exception as e:
            log.warning("fear_greed_history failed: %s", e)
            return []

    # Optional: Glassnode
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

# =========================
# TA helpers
# =========================


def ta_rsi(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [None] * period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = float('inf') if avg_loss == 0 else (avg_gain / avg_loss)
        rsis.append(100.0 - (100.0 / (1.0 + rs)))
    return [x for x in rsis if x is not None]


def ta_stoch_rsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3) -> Tuple[List[float], List[float]]:
    rsi_series = ta_rsi(closes, rsi_period)
    if len(rsi_series) < stoch_period + max(k, d):
        return [], []
    stoch = []
    for i in range(stoch_period - 1, len(rsi_series)):
        window = rsi_series[i - stoch_period + 1:i + 1]
        low, high = min(window), max(window)
        val = 0.0 if high == low else (rsi_series[i] - low) / (high - low)
        stoch.append(val)

    def sma(series, n):
        out = []
        for i in range(n - 1, len(series)):
            out.append(sum(series[i - n + 1:i + 1]) / n)
        return out
    k_line = sma(stoch, k)
    d_line = sma(k_line, d)
    k_line = k_line[len(k_line) - len(d_line):]
    return k_line, d_line


def _sma(vals: List[float], window: int) -> Optional[float]:
    if len(vals) < window:
        return None
    return sum(vals[-window:]) / float(window)


def _median_abs(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return abs(s[mid]) if n % 2 else abs((s[mid - 1] + s[mid]) / 2.0)

# =========================
# Metric aggregation
# =========================


async def _global_with_fallbacks(dc: DataClient) -> Dict[str, Any]:
    # CoinGecko
    cg = await _safe(dc.coingecko_global(), None, "coingecko_global")
    if cg and cg.get("data"):
        return cg

    # CoinMarketCap (requires key)
    cmc = await _safe(dc.cmc_global(), None, "cmc_global")
    if cmc and cmc.get("data"):
        d = cmc["data"]
        total = float(d["quote"]["USD"]["total_market_cap"])
        btc_dom = float(d.get("btc_dominance", 0.0))
        eth_dom = 0.0
        q = await _safe(dc.cmc_quotes(["BTC", "ETH"]), None, "cmc_quotes_btc_eth")
        if q and q.get("data"):
            try:
                btc_mcap = float(q["data"]["BTC"]["quote"]
                                 ["USD"]["market_cap"])
                eth_mcap = float(q["data"]["ETH"]["quote"]
                                 ["USD"]["market_cap"])
                btc_dom = (btc_mcap / total) * 100.0 if total > 0 else btc_dom
                eth_dom = (eth_mcap / total) * 100.0 if total > 0 else 0.0
            except Exception:
                pass
        return {"data": {"total_market_cap": {"usd": total},
                         "market_cap_percentage": {"btc": btc_dom, "eth": eth_dom}}}

    # CoinPaprika
    pk = await _safe(dc.paprika_global(), None, "paprika_global")
    if pk and isinstance(pk, dict):
        total = float(pk.get("market_cap_usd", 0.0))
        btc_dom = float(pk.get("bitcoin_dominance_percentage", 0.0))
        return {"data": {"total_market_cap": {"usd": total},
                         "market_cap_percentage": {"btc": btc_dom, "eth": 0.0}}}

    # Fallback to zeros
    return {"data": {"total_market_cap": {"usd": 0.0},
                     "market_cap_percentage": {"btc": 0.0, "eth": 0.0}}}


async def _eth_btc_with_fallback(dc: DataClient) -> float:
    v = await _safe(dc.coingecko_eth_btc(), None, "coingecko_eth_btc")
    if v is not None and v > 0:
        return float(v)
    okx = await _safe(dc.okx_eth_btc_ratio(), None, "okx_eth_btc_ratio")
    return float(okx or 0.0)


async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    # global & structure
    cg = await _global_with_fallbacks(dc)
    ethbtc = await _eth_btc_with_fallback(dc)

    total_mcap = float(((cg or {}).get("data", {}).get(
        "total_market_cap", {}).get("usd", 0.0)))
    btc_pct = float(((cg or {}).get("data", {}).get(
        "market_cap_percentage", {}).get("btc", 0.0)))
    eth_pct = float(((cg or {}).get("data", {}).get(
        "market_cap_percentage", {}).get("eth", 0.0)))
    btc_mcap = total_mcap * (btc_pct / 100.0) if total_mcap else 0.0
    eth_mcap = total_mcap * (eth_pct / 100.0) if total_mcap else 0.0
    altcap_calc = max(total_mcap - btc_mcap - eth_mcap, 0.0)

    # Altcap via internal calculation only
    altcap_mcap = altcap_calc
    altcap_btc_ratio = (altcap_mcap / btc_mcap) if btc_mcap > 0 else 0.0

    # funding basket (OKX)
    funding: Dict[str, Dict[str, Any]] = {}
    abs_funding_list: List[float] = []
    for sym in CFG.symbols["funding"]:
        inst = to_okx_instId(sym)
        fr = await _safe(dc.okx_funding_rate(inst), 0.0, f"funding {inst}")
        lastp = await _safe(dc.okx_ticker_last_price(inst), 0.0, f"lastPrice {inst}")
        funding[sym] = {"lastFundingRate": fr or 0.0,
                        "markPrice": lastp or 0.0}
        abs_funding_list.append(abs((fr or 0.0) * 100.0))
    max_fr = 0.0
    max_sym = None
    for sym, d in funding.items():
        v = abs(float(d.get("lastFundingRate", 0.0)) * 100.0)
        if v > max_fr:
            max_fr, max_sym = v, sym
    median_fr = _median_abs(abs_funding_list) or 0.0
    top3 = sorted(((s, abs(d.get("lastFundingRate", 0.0)) * 100.0) for s, d in funding.items()),
                  key=lambda x: x[1], reverse=True)[:3]

    # open interest USD (OKX)
    oi_usd: Dict[str, Optional[float]] = {}
    for sym in CFG.symbols["oi"]:
        inst = to_okx_instId(sym)
        uly = to_okx_uly(sym)
        notional = await _safe(dc.okx_open_interest_usd_by_inst(inst), None, f"oi {inst}")
        if notional is None:
            notional = await _safe(dc.okx_open_interest_usd_by_uly(uly), None, f"oi {uly}")
        oi_usd[sym] = notional

    # sentiment
    trends_avg, trends_by_kw = await _safe(
        dc.google_trends_score(["crypto", "bitcoin", "ethereum"]),
        (0.0, {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0}),
        "google_trends_score"
    )
    fng_value, fng_label = await _safe(dc.fear_greed(), (None, None), "fear_greed")
    fng_hist = await _safe(dc.fear_greed_history(60), [], "fng_history")

    def mean(arr: List[float]) -> Optional[float]:
        return (sum(arr) / len(arr)) if arr else None

    greed_thr = float(CFG.thresholds.get("fear_greed_greed_min", 70))
    fng_ma14 = mean([v for v in fng_hist[:14]]) if len(
        fng_hist) >= 14 else None
    fng_ma30 = mean([v for v in fng_hist[:30]]) if len(
        fng_hist) >= 30 else None
    greed_streak = 0
    for v in fng_hist:
        if v >= greed_thr:
            greed_streak += 1
        else:
            break
    greed_pct_30d = (sum(
        1 for v in fng_hist[:30] if v >= greed_thr) / 30.0) if len(fng_hist) >= 30 else None

    # on-chain & pi cycle
    mvrv_z = await _safe(dc.glassnode_mvrv_z("BTC"), None, "glassnode_mvrv_z")
    exch_inflow = await _safe(dc.glassnode_exchange_inflow("BTC"), None, "glassnode_exchange_inflow")

    # --- Pi Cycle inputs: prefer CoinGecko daily; fallback to OKX spot daily
    btc_daily: List[float] = await _safe(dc.coingecko_btc_daily("max"), [], "btc_daily")
    if len(btc_daily) < 350:
        if not btc_daily:
            btc_daily = await _safe(dc.coingecko_btc_daily(1500), btc_daily, "btc_daily1500")
        if len(btc_daily) < 350:
            btc_daily = await _safe(dc.coingecko_btc_daily(500), btc_daily, "btc_daily500")
    if len(btc_daily) < 350:
        btc_daily = await _safe(dc.okx_daily_closes("BTC-USDT", 500), [], "btc_daily_okx")

    sma111 = _sma(btc_daily, 111) if btc_daily else None
    sma350 = _sma(btc_daily, 350) if btc_daily else None
    two_350 = (2.0 * sma350) if (sma350 is not None) else None
    pi_ratio = (sma111 / two_350) if (sma111 and two_350) else None

    # weekly momentum & fib (OKX weekly candles)
    btc_w = await _safe(dc.okx_weekly_closes("BTC-USDT"), [], "btc_weekly")
    ethbtc_w = await _safe(dc.okx_weekly_closes("ETH-BTC"), [], "ethbtc_weekly")
    btc_rsi_w = ta_rsi(btc_w, 14)
    ethbtc_rsi_w = ta_rsi(ethbtc_w, 14)
    btc_k, btc_d = ta_stoch_rsi(btc_w, 14, 14, 3, 3)
    btc_rsi_val = btc_rsi_w[-1] if btc_rsi_w else None
    ethbtc_rsi_val = ethbtc_rsi_w[-1] if ethbtc_rsi_w else None
    btc_k_last = btc_k[-1] if btc_k else None
    btc_d_last = btc_d[-1] if btc_d else None

    def find_swing_low_high(vals: List[float], lookback: int = 160) -> Tuple[Optional[float], Optional[float]]:
        if len(vals) < 20:
            return None, None
        sub = vals[-lookback:] if len(vals) > lookback else vals[:]
        return (min(sub), max(sub))

    def fib_extensions(low: float, high: float, levels=(1.272, 1.414, 1.618)) -> Dict[str, float]:
        diff = high - low
        return {f"{lvl:.3f}": high + diff * (lvl - 1.0) for lvl in levels}

    def proximity_pct(price: float, target: float) -> float:
        return abs(price - target) / target if target else 1.0

    # ===== Equal-weight ALT basket momentum (weekly) =====
    alt_ids = CFG.symbols.get("alt_basket", [])
    alt_series: Dict[str, List[float]] = {}
    for inst in alt_ids:
        closes = await _safe(dc.okx_weekly_closes(inst), [], f"alt_weekly_{inst}")
        if closes:
            alt_series[inst] = closes

    alt_index_series: List[float] = []
    if len(alt_series) >= 2:
        # Align by shortest history
        min_len = min(len(v) for v in alt_series.values())
        if min_len >= 20:
            norm = []
            for inst, seq in alt_series.items():
                s = seq[-min_len:]
                base = s[0]
                if base <= 0:
                    continue
                norm.append([x / base for x in s])
            if len(norm) >= 2:
                # equal-weight average
                for i in range(min_len):
                    alt_index_series.append(
                        sum(series[i] for series in norm) / len(norm))

    alt_rsi_val = None
    alt_k_last = None
    alt_d_last = None
    alt_fib = {}
    alt_fib_near = None
    alt_fib_near_pct = None

    if len(alt_index_series) >= 30:
        alt_rsi = ta_rsi(alt_index_series, 14)
        alt_k, alt_d = ta_stoch_rsi(alt_index_series, 14, 14, 3, 3)
        alt_rsi_val = alt_rsi[-1] if alt_rsi else None
        alt_k_last = alt_k[-1] if alt_k else None
        alt_d_last = alt_d[-1] if alt_d else None

        low_alt, high_alt = find_swing_low_high(alt_index_series, 160)
        last_alt = alt_index_series[-1]
        if low_alt is not None and high_alt is not None:
            alt_fib = fib_extensions(low_alt, high_alt)
            best_lvl, best_pct = None, None
            for name, lvl in alt_fib.items():
                p = proximity_pct(last_alt, lvl)
                if best_pct is None or p < best_pct:
                    best_pct, best_lvl = p, name
            alt_fib_near, alt_fib_near_pct = best_lvl, best_pct

    # BTC fib
    btc_low, btc_high = find_swing_low_high(btc_w, 160)
    fib = {}
    fib_nearest = None
    fib_nearest_pct = None
    btc_last = btc_w[-1] if btc_w else None
    if btc_low is not None and btc_high is not None and btc_last is not None:
        fib = fib_extensions(btc_low, btc_high)
        best_lvl, best_pct = None, None
        for name, lvl in fib.items():
            p = proximity_pct(btc_last, lvl)
            if best_pct is None or p < best_pct:
                best_pct, best_lvl = p, name
        fib_nearest, fib_nearest_pct = best_lvl, best_pct

    momentum = {
        "btc_weekly_rsi": btc_rsi_val,
        "ethbtc_weekly_rsi": ethbtc_rsi_val,
        "btc_weekly_stoch_k": btc_k_last,
        "btc_weekly_stoch_d": btc_d_last,
        "fib_ext": fib,
        "fib_nearest": fib_nearest,
        "fib_nearest_pct": fib_nearest_pct,
        # ALT basket
        "alt_weekly_rsi": alt_rsi_val,
        "alt_weekly_stoch_k": alt_k_last,
        "alt_weekly_stoch_d": alt_d_last,
        "alt_fib_ext": alt_fib,
        "alt_fib_nearest": alt_fib_near,
        "alt_fib_nearest_pct": alt_fib_near_pct,
    }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # market structure
        "btc_dominance_pct": btc_pct,
        "eth_btc": ethbtc,
        "altcap_btc_ratio": altcap_btc_ratio,

        # derivatives
        "funding": funding,
        "funding_stats": {
            "max_abs_pct": max_fr,
            "max_sym": max_sym,
            "median_abs_pct": median_fr,
            "top3_abs_pct": top3,
        },
        "open_interest_usd": oi_usd,

        # sentiment
        "google_trends_avg7d": float(trends_avg or 0.0),
        "google_trends_breakdown": trends_by_kw or {"crypto": 0.0, "bitcoin": 0.0, "ethereum": 0.0},
        "fear_greed_index": fng_value,
        "fear_greed_label": fng_label,
        "fear_greed_ma14": fng_ma14,
        "fear_greed_ma30": fng_ma30,
        "fear_greed_greed_streak_days": greed_streak,
        "fear_greed_pct30d_greed": greed_pct_30d,

        # cycle/on-chain
        "mvrv_z": mvrv_z,
        "exchange_inflow_proxy": exch_inflow,
        "pi_cycle": {"sma111": sma111, "sma350x2": two_350, "ratio": pi_ratio},

        # momentum & fib
        "momentum": momentum,
    }

# =========================
# Composite score
# =========================


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def compute_alt_top_certainty(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
    subs: Dict[str, float] = {}

    def frac(value: Optional[float], threshold: Optional[float]) -> float:
        if value is None or threshold is None or threshold <= 0:
            return 0.0
        return clamp01(float(value) / float(threshold))

    # structure
    subs["altcap_vs_btc"] = frac(
        m["altcap_btc_ratio"], t["altcap_btc_ratio_max"])
    subs["eth_btc"] = frac(m["eth_btc"],           t["eth_btc_ratio_max"])
    subs["btc_dominance"] = frac(
        m["btc_dominance_pct"], t["btc_dominance_max"])

    # derivatives
    max_fr_abs = float(m.get("funding_stats", {}).get("max_abs_pct", 0.0))
    subs["funding_extreme"] = frac(max_fr_abs, t["funding_rate_abs_max"])
    oi = m.get("open_interest_usd") or {}
    subs["oi_btc"] = frac(
        oi.get("BTCUSDT"), t["open_interest_btc_usdt_usd_max"])
    subs["oi_eth"] = frac(
        oi.get("ETHUSDT"), t["open_interest_eth_usdt_usd_max"])

    # sentiment
    subs["trends"] = frac(m["google_trends_avg7d"],
                          t["google_trends_7d_avg_min"])
    fg_val = m.get("fear_greed_index")
    subs["fear_greed"] = frac(float(fg_val) if fg_val is not None else None,
                              float(t.get("fear_greed_greed_min", 70)))
    ma14 = m.get("fear_greed_ma14")
    ma30 = m.get("fear_greed_ma30")
    streak = m.get("fear_greed_greed_streak_days")
    pct30 = m.get("fear_greed_pct30d_greed")

    def above_ratio(val, warn, flag):
        if val is None:
            return 0.0
        if val >= flag:
            return 1.0
        if val <= warn:
            return 0.0
        return (val - warn) / (flag - warn)
    subs["fng_ma14"] = above_ratio(ma14, t.get(
        "fear_greed_ma14_min", 70), t.get("fear_greed_ma14_min", 70))
    subs["fng_ma30"] = above_ratio(ma30, t.get(
        "fear_greed_ma30_min", 65), t.get("fear_greed_ma30_min", 65))

    def frac_persist(val, thr):
        if val is None or thr is None:
            return 0.0
        return clamp01(val / thr)
    p_days = frac_persist(streak, t.get("greed_persistence_days", 10))
    p_pct = frac_persist(pct30,  t.get("greed_persistence_pct30_min", 0.60))
    subs["fng_persist"] = max(p_days, p_pct)

    # cycle & momentum
    pi_ratio = (m.get("pi_cycle") or {}).get("ratio")
    subs["pi_cycle"] = frac(pi_ratio, t.get("pi_cycle_ratio_min", 0.98))
    mz = m.get("mvrv_z")
    subs["mvrv_z"] = frac(
        mz, float(t.get("mvrv_z_extreme", 7.0))) if mz is not None else 0.0

    mom = m.get("momentum", {}) if isinstance(m.get("momentum"), dict) else {}
    rsi_btc = mom.get("btc_weekly_rsi")
    rsi_ethb = mom.get("ethbtc_weekly_rsi")
    k_last = mom.get("btc_weekly_stoch_k")
    d_last = mom.get("btc_weekly_stoch_d")
    fib_pct = mom.get("fib_nearest_pct")
    # ALT basket
    rsi_alt = mom.get("alt_weekly_rsi")
    k_alt = mom.get("alt_weekly_stoch_k")
    d_alt = mom.get("alt_weekly_stoch_d")
    fib_alt_pct = mom.get("alt_fib_nearest_pct")

    subs["rsi_btc_w"] = above_ratio(rsi_btc,  t.get(
        "btc_weekly_rsi_warn", 60),  t.get("btc_weekly_rsi_flag", 70))
    subs["rsi_ethbtc_w"] = above_ratio(rsi_ethb, t.get(
        "ethbtc_weekly_rsi_warn", 55), t.get("ethbtc_weekly_rsi_flag", 65))

    def stoch_cross_risk(kv, dv, ob=0.80):
        if kv is None or dv is None:
            return 0.0
        return 1.0 if (kv < dv and max(kv, dv) >= ob) else (0.5 if max(kv, dv) >= ob else 0.0)

    subs["stoch_btc_w"] = stoch_cross_risk(
        k_last, d_last, t.get("stochrsi_overbought", 0.80))

    def fib_risk(pct, warn=0.03, flag=0.015):
        if pct is None:
            return 0.0
        if pct <= flag:
            return 1.0
        if pct >= warn:
            return 0.0
        return (warn - pct) / (warn - flag)

    subs["fib_btc"] = fib_risk(fib_pct, t.get(
        "fib_warn_pct", 0.03), t.get("fib_flag_pct", 0.015))

    # ALT basket subs
    subs["rsi_alt_w"] = above_ratio(rsi_alt, t.get(
        "alt_weekly_rsi_warn", 65), t.get("alt_weekly_rsi_flag", 75))
    subs["stoch_alt_w"] = stoch_cross_risk(
        k_alt, d_alt, t.get("stochrsi_overbought", 0.80))
    subs["fib_alt"] = fib_risk(fib_alt_pct, t.get(
        "fib_warn_pct", 0.03), t.get("fib_flag_pct", 0.015))

    # weights (normalized by sum)
    weights = {
        "altcap_vs_btc": 0.12, "eth_btc": 0.10, "btc_dominance": 0.05,
        "funding_extreme": 0.10, "oi_btc": 0.08, "oi_eth": 0.06,
        "trends": 0.05, "fear_greed": 0.05, "fng_ma14": 0.04, "fng_ma30": 0.04, "fng_persist": 0.06,
        "pi_cycle": 0.04, "mvrv_z": 0.04,
        "rsi_btc_w": 0.06, "rsi_ethbtc_w": 0.06, "stoch_btc_w": 0.06, "fib_btc": 0.06,
        "rsi_alt_w": 0.07, "stoch_alt_w": 0.07, "fib_alt": 0.07,
    }
    total_w = sum(weights.values())
    score01 = sum(subs.get(k, 0.0) * w for k, w in weights.items()
                  ) / total_w if total_w > 0 else 0.0
    return int(round(score01 * 100)), subs

# =========================
# Presentation
# =========================


def severity_above(value: Optional[float], thr: Optional[float], warn_frac: float) -> str:
    if value is None or thr is None or thr <= 0:
        return "green"
    v = float(value)
    if v >= float(thr):
        return "red"
    if v >= float(thr) * float(warn_frac):
        return "yellow"
    return "green"


def severity_score(score: int, warn: int, flag: int) -> str:
    if score >= flag:
        return "red"
    if score >= warn:
        return "yellow"
    return "green"


def bullet3(sev: str) -> str:
    return "🔴" if sev == "red" else "🟡" if sev == "yellow" else "🟢"


def b(txt: str) -> str:
    return f"<b>{txt}</b>"


def fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def build_status_text(m: Dict[str, Any], t: Dict[str, Any], profile_name: str = "") -> str:
    warn_frac = float(t.get("warn_fraction", 0.8))

    # Market structure
    dom = m["btc_dominance_pct"]
    sev_dom = severity_above(dom, t["btc_dominance_max"], warn_frac)

    ethbtc = m["eth_btc"]
    sev_ethbtc = severity_above(ethbtc, t["eth_btc_ratio_max"], warn_frac)

    altbtc = m["altcap_btc_ratio"]
    sev_altbtc = severity_above(altbtc, t["altcap_btc_ratio_max"], warn_frac)

    # Derivatives
    fstats = m.get("funding_stats", {})
    max_fr = float(fstats.get("max_abs_pct", 0.0))
    sev_fund = severity_above(max_fr, t["funding_rate_abs_max"], warn_frac)
    median_fr = float(fstats.get("median_abs_pct", 0.0))
    sev_fund_median = severity_above(
        median_fr, t["funding_rate_abs_max"], warn_frac)
    top3 = fstats.get("top3_abs_pct", []) or []
    oi = m.get("open_interest_usd", {})
    btc_oi = oi.get("BTCUSDT")
    eth_oi = oi.get("ETHUSDT")
    sev_btc_oi = severity_above(
        btc_oi, t["open_interest_btc_usdt_usd_max"], warn_frac)
    sev_eth_oi = severity_above(
        eth_oi, t["open_interest_eth_usdt_usd_max"], warn_frac)

    # Sentiment
    gavg = m["google_trends_avg7d"]
    sev_trends = severity_above(gavg, t["google_trends_7d_avg_min"], warn_frac)
    fng_val = m.get("fear_greed_index")
    sev_fng_now = severity_above(float(fng_val) if fng_val is not None else None,
                                 t.get("fear_greed_greed_min", 70), warn_frac)
    fng_ma14 = m.get("fear_greed_ma14")
    fng_ma30 = m.get("fear_greed_ma30")
    streak = int(m.get("fear_greed_greed_streak_days") or 0)
    pct30 = m.get("fear_greed_pct30d_greed")
    sev_ma14 = severity_above(fng_ma14, t.get(
        "fear_greed_ma14_min", 70), warn_frac)
    sev_ma30 = severity_above(fng_ma30, t.get(
        "fear_greed_ma30_min", 65), warn_frac)
    sev_streak = severity_above(streak, t.get(
        "greed_persistence_days", 10), warn_frac)
    sev_pct30 = severity_above(pct30, t.get(
        "greed_persistence_pct30_min", 0.60), warn_frac)
    sev_persist = "red" if (sev_streak == "red" or sev_pct30 == "red") else (
        "yellow" if (sev_streak == "yellow" or sev_pct30 == "yellow") else "green")

    # Cycle / On-chain
    pi = m.get("pi_cycle") or {}
    pi_ratio = pi.get("ratio")
    sev_pi = severity_above(pi_ratio, t.get(
        "pi_cycle_ratio_min", 0.98), warn_frac)
    mz_val = m.get("mvrv_z")
    sev_mz = severity_above(mz_val, t.get("mvrv_z_extreme", 7.0), warn_frac)

    # Momentum & Fib
    mom = m.get("momentum", {}) or {}
    rsi_btc = mom.get("btc_weekly_rsi")
    rsi_ethb = mom.get("ethbtc_weekly_rsi")
    k_last = mom.get("btc_weekly_stoch_k")
    d_last = mom.get("btc_weekly_stoch_d")
    fib_near = mom.get("fib_nearest")
    fib_pct = mom.get("fib_nearest_pct")

    # ALT basket
    rsi_alt = mom.get("alt_weekly_rsi")
    k_alt = mom.get("alt_weekly_stoch_k")
    d_alt = mom.get("alt_weekly_stoch_d")
    alt_fib_near = mom.get("alt_fib_nearest")
    alt_fib_pct = mom.get("alt_fib_nearest_pct")

    overb = t.get("stochrsi_overbought", 0.80)
    stoch_sev = "red" if (k_last is not None and d_last is not None and k_last < d_last and max(k_last, d_last) >= overb) \
        else ("yellow" if (max(k_last or 0, d_last or 0) >= overb) else "green")
    alt_stoch_sev = "red" if (k_alt is not None and d_alt is not None and k_alt < d_alt and max(k_alt, d_alt) >= overb) \
        else ("yellow" if (max(k_alt or 0, d_alt or 0) >= overb) else "green")

    fib_flag_pct = t.get("fib_flag_pct", 0.015)
    fib_warn_pct = t.get("fib_warn_pct", 0.03)
    if fib_pct is None:
        sev_fib = "green"
    else:
        sev_fib = "red" if fib_pct < fib_flag_pct else (
            "yellow" if fib_pct < fib_warn_pct else "green")
    if alt_fib_pct is None:
        alt_sev_fib = "green"
    else:
        alt_sev_fib = "red" if alt_fib_pct < fib_flag_pct else (
            "yellow" if alt_fib_pct < fib_warn_pct else "green")

    # Precompute display thresholds (avoid nested f-strings)
    warn_dom = t['warn_fraction'] * t['btc_dominance_max']
    warn_ethbtc = t['warn_fraction'] * t['eth_btc_ratio_max']
    warn_altbtc = t['warn_fraction'] * t['altcap_btc_ratio_max']
    warn_fund = t['warn_fraction'] * t['funding_rate_abs_max']
    warn_oi_btc = t['warn_fraction'] * t['open_interest_btc_usdt_usd_max']
    warn_oi_eth = t['warn_fraction'] * t['open_interest_eth_usdt_usd_max']
    warn_trends = t['warn_fraction'] * t['google_trends_7d_avg_min']
    warn_fng14 = int(t['warn_fraction'] * t.get('fear_greed_ma14_min', 70))
    warn_fng30 = int(t['warn_fraction'] * t.get('fear_greed_ma30_min', 65))
    warn_days = int(t['warn_fraction'] * t.get('greed_persistence_days', 10))
    warn_pct = int(100 * t['warn_fraction'] *
                   t.get('greed_persistence_pct30_min', 0.60))
    flag_pct = int(100 * t.get('greed_persistence_pct30_min', 0.60))
    pi_thr_pct = t.get('pi_cycle_ratio_min', 0.98) * 100.0
    warn_pi_pct = t['warn_fraction'] * pi_thr_pct

    lines: List[str] = []
    lines.append(f"📊 {b('Crypto Market Snapshot')} — {m['timestamp']} UTC")
    if profile_name:
        lines.append(f"Profile: {b(profile_name)}")
    lines.append("")

    # Market Structure
    lines.append(b("Market Structure"))
    lines.append(f"• Bitcoin market share of total crypto: {bullet3(sev_dom)} {b(f'{dom:.2f}%')}  "
                 f"(warn ≥ {warn_dom:.2f}%, flag ≥ {t['btc_dominance_max']:.2f}%)")
    lines.append(f"• Ether price relative to Bitcoin (ETH/BTC): {bullet3(sev_ethbtc)} {b(f'{ethbtc:.5f}')}  "
                 f"(warn ≥ {warn_ethbtc:.5f}, flag ≥ {t['eth_btc_ratio_max']:.5f})")
    lines.append(f"• Altcoin market cap / Bitcoin market cap: {bullet3(sev_altbtc)} {b(f'{altbtc:.2f}')}  "
                 f"(warn ≥ {warn_altbtc:.2f}, flag ≥ {t['altcap_btc_ratio_max']:.2f})")
    lines.append("")

    # Derivatives
    lines.append(b("Derivatives"))
    sym_list = ", ".join(CFG.symbols.get("funding", []))
    t3 = ", ".join([f"{s} {v:.3f}%" for s, v in top3]) if top3 else "n/a"
    lines.append(f"• Funding (basket: {sym_list}) — max: {bullet3(sev_fund)} {b(f'{max_fr:.3f}%')} | "
                 f"median: {bullet3(sev_fund_median)} {b(f'{median_fr:.3f}%')}  "
                 f"(warn ≥ {warn_fund:.3f}%, flag ≥ {t['funding_rate_abs_max']:.3f}%)")
    lines.append(f"  Top-3 funding extremes: {b(t3)}")
    lines.append(f"• Bitcoin open interest (USD): {bullet3(sev_btc_oi)} {b(fmt_usd(btc_oi))}  "
                 f"(warn ≥ {fmt_usd(warn_oi_btc)}, flag ≥ {fmt_usd(t['open_interest_btc_usdt_usd_max'])})")
    lines.append(f"• Ether open interest (USD): {bullet3(sev_eth_oi)} {b(fmt_usd(eth_oi))}  "
                 f"(warn ≥ {fmt_usd(warn_oi_eth)}, flag ≥ {fmt_usd(t['open_interest_eth_usdt_usd_max'])})")
    lines.append("")

    # Sentiment
    lines.append(b("Sentiment"))
    lines.append(f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {bullet3(sev_trends)} {b(f'{gavg:.1f}')}  "
                 f"(warn ≥ {warn_trends:.1f}, flag ≥ {t['google_trends_7d_avg_min']:.1f})")
    if fng_val is not None:
        lines.append(f"• Fear & Greed Index (overall crypto): {bullet3(sev_fng_now)} "
                     f"{b(str(fng_val))} ({m.get('fear_greed_label') or 'n/a'})  "
                     f"(warn ≥ {int(t['warn_fraction']*t.get('fear_greed_greed_min',70))}, "
                     f"flag ≥ {t.get('fear_greed_greed_min',70)})")
    else:
        lines.append("• Fear & Greed Index: 🟢 " + b("n/a"))
    lines.append(f"• Fear & Greed 14-day average: {bullet3(sev_ma14)} "
                 f"{b('n/a' if fng_ma14 is None else f'{fng_ma14:.1f}')}  "
                 f"(warn ≥ {warn_fng14}, flag ≥ {t.get('fear_greed_ma14_min',70)})")
    lines.append(f"• Fear & Greed 30-day average: {bullet3(sev_ma30)} "
                 f"{b('n/a' if fng_ma30 is None else f'{fng_ma30:.1f}')}  "
                 f"(warn ≥ {warn_fng30}, flag ≥ {t.get('fear_greed_ma30_min',65)})")
    persist_left = f"{streak} days in a row"
    persist_right_val = "n/a" if pct30 is None else f"{pct30*100:.0f}% of last 30 days ≥ {int(t.get('fear_greed_greed_min',70))}"
    lines.append(f"• Greed persistence: {bullet3(sev_persist)} {b(persist_left)} | {b(persist_right_val)}  "
                 f"(warn: days ≥ {warn_days} or pct ≥ {warn_pct}%; "
                 f"flag: days ≥ {t.get('greed_persistence_days',10)} or pct ≥ {flag_pct}%)")
    lines.append("")

    # Cycle & On-Chain
    lines.append(b("Cycle & On-Chain"))
    if pi_ratio is not None and pi.get("sma111") and pi.get("sma350x2"):
        lines.append(
            f"• Pi Cycle Top proximity (111-DMA vs 2×350-DMA): {bullet3(sev_pi)} "
            f"{b(f'{pi_ratio*100.0:.1f}% of cross')} "
            f"(111DMA {fmt_usd(pi['sma111'])}, 2×350DMA {fmt_usd(pi['sma350x2'])}; "
            f"warn ≥ {warn_pi_pct:.1f}%, flag ≥ {pi_thr_pct:.1f}%)"
        )
    else:
        lines.append("• Pi Cycle Top proximity: 🟢 " + b("n/a"))
    if mz_val is not None:
        lines.append(f"• Bitcoin MVRV Z-Score: {bullet3(sev_mz)} {b(f'{mz_val:.2f}')}  "
                     f"(warn ≥ {t['warn_fraction']*t['mvrv_z_extreme']:.2f}, "
                     f"flag ≥ {t['mvrv_z_extreme']:.2f})")
    lines.append("")

    # Momentum & Extensions (Weekly)
    lines.append(b("Momentum & Extensions (Weekly)"))
    sev_rsi_btc = severity_above(rsi_btc,  t.get(
        "btc_weekly_rsi_flag", 70), t.get("warn_fraction", 0.8))
    sev_rsi_ethb = severity_above(rsi_ethb, t.get(
        "ethbtc_weekly_rsi_flag", 65), t.get("warn_fraction", 0.8))
    lines.append(f"• BTC RSI (1W): {bullet3(sev_rsi_btc)} {b('n/a' if rsi_btc is None else f'{rsi_btc:.1f}')} "
                 f"(warn ≥ {t.get('btc_weekly_rsi_warn',60):.1f}, flag ≥ {t.get('btc_weekly_rsi_flag',70):.1f})")
    lines.append(f"• ETH/BTC RSI (1W): {bullet3(sev_rsi_ethb)} {b('n/a' if rsi_ethb is None else f'{rsi_ethb:.1f}')} "
                 f"(warn ≥ {t.get('ethbtc_weekly_rsi_warn',55):.1f}, flag ≥ {t.get('ethbtc_weekly_rsi_flag',65):.1f})")
    stoch_val = "n/a" if (
        k_last is None or d_last is None) else f"{k_last:.2f}/{d_last:.2f}"
    lines.append(f"• BTC Stoch RSI (1W) K/D: {bullet3(stoch_sev)} {b(stoch_val)} "
                 f"(overbought ≥ {overb:.2f}; red = bearish cross from OB)")
    fib_val = "n/a" if (
        fib_near is None or fib_pct is None) else f"{fib_near} @ {fib_pct*100:.2f}% away"
    lines.append(f"• BTC Fibonacci extension proximity: {bullet3(sev_fib)} {b(fib_val)} "
                 f"(warn ≤ {fib_warn_pct*100:.1f}%, flag ≤ {fib_flag_pct*100:.1f}%)")

    # ALT basket
    sev_rsi_alt = severity_above(rsi_alt, t.get(
        "alt_weekly_rsi_flag", 75), t.get("warn_fraction", 0.8))
    alt_stoch_val = "n/a" if (
        k_alt is None or d_alt is None) else f"{k_alt:.2f}/{d_alt:.2f}"
    alt_fib_val = "n/a" if (
        alt_fib_near is None or alt_fib_pct is None) else f"{alt_fib_near} @ {alt_fib_pct*100:.2f}% away"
    lines.append(f"• ALT basket RSI (1W, equal-weight): {bullet3(sev_rsi_alt)} "
                 f"{b('n/a' if rsi_alt is None else f'{rsi_alt:.1f}')} "
                 f"(warn ≥ {t.get('alt_weekly_rsi_warn',65):.1f}, flag ≥ {t.get('alt_weekly_rsi_flag',75):.1f})")
    lines.append(f"• ALT basket Stoch RSI (1W) K/D: {bullet3(alt_stoch_sev)} {b(alt_stoch_val)} "
                 f"(overbought ≥ {overb:.2f}; red = bearish cross from OB)")
    lines.append(f"• ALT basket Fibonacci proximity: {bullet3(alt_sev_fib)} {b(alt_fib_val)} "
                 f"(warn ≤ {fib_warn_pct*100:.1f}%, flag ≤ {fib_flag_pct*100:.1f}%)")
    lines.append("")

    # Composite
    certainty, subs = compute_alt_top_certainty(m, t)
    comp_warn = int(t.get("composite_warn", 40))
    comp_flag = int(t.get("composite_flag", 70))
    comp_sev = severity_score(certainty, comp_warn, comp_flag)

    lines.append(b("Alt-Top Certainty (Composite)"))
    lines.append(f"• Certainty: {bullet3(comp_sev)} {b(f'{certainty}/100')} "
                 f"(yellow ≥ {comp_warn}, red ≥ {comp_flag})")
    pretty_sub = [
        f"altcap_vs_btc {int(round(subs.get('altcap_vs_btc',0)*100))}%",
        f"eth_btc {int(round(subs.get('eth_btc',0)*100))}%",
        f"funding {int(round(subs.get('funding_extreme',0)*100))}%",
        f"OI_BTC {int(round(subs.get('oi_btc',0)*100))}%",
        f"OI_ETH {int(round(subs.get('oi_eth',0)*100))}%",
        f"trends {int(round(subs.get('trends',0)*100))}%",
        f"F&G {int(round(subs.get('fear_greed',0)*100))}%",
        f"F&G14 {int(round(subs.get('fng_ma14',0)*100))}%",
        f"F&G30 {int(round(subs.get('fng_ma30',0)*100))}%",
        f"F&GPersist {int(round(subs.get('fng_persist',0)*100))}%",
        f"Pi {int(round(subs.get('pi_cycle',0)*100))}%",
        f"MVRVZ {int(round(subs.get('mvrv_z',0)*100))}%",
        f"RSI_BTC_W {int(round(subs.get('rsi_btc_w',0)*100))}%",
        f"RSI_ETHBTC_W {int(round(subs.get('rsi_ethbtc_w',0)*100))}%",
        f"Stoch_BTC_W {int(round(subs.get('stoch_btc_w',0)*100))}%",
        f"Fib_BTC {int(round(subs.get('fib_btc',0)*100))}%",
        f"RSI_ALT_W {int(round(subs.get('rsi_alt_w',0)*100))}%",
        f"Stoch_ALT_W {int(round(subs.get('stoch_alt_w',0)*100))}%",
        f"Fib_ALT {int(round(subs.get('fib_alt',0)*100))}%",
    ]
    lines.append("• Subscores: " + ", ".join(pretty_sub))

    return "\n".join(lines)

# =========================
# Flags
# =========================


def evaluate_flags(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[str]]:
    flags: List[str] = []
    if m["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("High Bitcoin dominance")
    if m["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("Elevated ETH/BTC ratio")
    if m["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Alt market cap stretched vs BTC")
    max_fr = float(m.get("funding_stats", {}).get("max_abs_pct", 0.0))
    if max_fr >= t["funding_rate_abs_max"]:
        t3 = m.get("funding_stats", {}).get("top3_abs_pct", [])
        flags.append(
            f"Perpetual funding extreme ({t3[0][0]} {max_fr:.3f}%)" if t3 else f"Perpetual funding extreme ({max_fr:.3f}%)")
    oi = m.get("open_interest_usd") or {}
    if oi.get("BTCUSDT") and oi["BTCUSDT"] >= t["open_interest_btc_usdt_usd_max"]:
        flags.append(f"High Bitcoin open interest (${oi['BTCUSDT']:,.0f})")
    if oi.get("ETHUSDT") and oi["ETHUSDT"] >= t["open_interest_eth_usdt_usd_max"]:
        flags.append(f"High Ether open interest (${oi['ETHUSDT']:,.0f})")
    if m["google_trends_avg7d"] >= t["google_trends_7d_avg_min"]:
        flags.append("Elevated retail interest (Google Trends)")
    fg = m.get("fear_greed_index")
    if fg is not None and fg >= t.get("fear_greed_greed_min", 70):
        flags.append("Greed is elevated (Fear & Greed Index)")
    if m.get("fear_greed_ma14") is not None and m["fear_greed_ma14"] >= t.get("fear_greed_ma14_min", 70):
        flags.append("Fear & Greed 14-day avg in Greed")
    if m.get("fear_greed_ma30") is not None and m["fear_greed_ma30"] >= t.get("fear_greed_ma30_min", 65):
        flags.append("Fear & Greed 30-day avg in Greed")
    streak = int(m.get("fear_greed_greed_streak_days") or 0)
    if streak >= t.get("greed_persistence_days", 10):
        flags.append(
            f"Greed streak {streak}d (≥{t.get('greed_persistence_days',10)}d)")
    pct30 = m.get("fear_greed_pct30d_greed")
    if pct30 is not None and pct30 >= t.get("greed_persistence_pct30_min", 0.60):
        flags.append(f"Greed in {int(pct30*100)}% of last 30d")
    pi = m.get("pi_cycle") or {}
    if pi.get("ratio") is not None and pi["ratio"] >= t.get("pi_cycle_ratio_min", 0.98):
        flags.append("Pi Cycle Top proximity elevated")
    if m.get("mvrv_z") is not None and t.get("mvrv_z_extreme") is not None and m["mvrv_z"] >= t["mvrv_z_extreme"]:
        flags.append("On-chain overvaluation (MVRV Z-Score)")

    # Weekly momentum cluster (BTC)
    mom = m.get("momentum", {}) or {}
    k = mom.get("btc_weekly_stoch_k")
    d = mom.get("btc_weekly_stoch_d")
    ob = t.get("stochrsi_overbought", 0.80)
    if k is not None and d is not None and k < d and max(k, d) >= ob:
        flags.append("Weekly Stoch RSI bearish cross from overbought (BTC)")

    # Weekly momentum cluster (ALT basket)
    k_alt = mom.get("alt_weekly_stoch_k")
    d_alt = mom.get("alt_weekly_stoch_d")
    if k_alt is not None and d_alt is not None and k_alt < d_alt and max(k_alt, d_alt) >= ob:
        flags.append(
            "Weekly Stoch RSI bearish cross from overbought (ALT basket)")

    return len(flags), flags

# =========================
# Telegram Bot
# =========================


SUBSCRIBERS: set[int] = set()


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    msg = (
        "👋 Crypto Cycle Watch online.\n\n"
        "Commands:\n"
        "• /status – snapshot\n"
        "• /assess – snapshot + flags\n"
        "• /assess_json [pretty|compact|file] – raw JSON\n"
        "• /subscribe – daily summary & alerts here\n"
        "• /unsubscribe – stop messages\n"
        "• /risk &lt;conservative|moderate|aggressive&gt;\n"
        "• /getrisk – show profile\n"
        "• /settime HH:MM – daily summary time (UTC)\n"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        t = get_thresholds_for_chat(update.effective_chat.id)
        profile = get_profile_for_chat(update.effective_chat.id)
        text = build_status_text(m, t, profile)
        await update.message.reply_text(text, parse_mode="HTML")
    finally:
        await dc.close()


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Could not fetch metrics right now: {e}", parse_mode="HTML")
        await dc.close()
        return
    try:
        t = get_thresholds_for_chat(chat_id)
        profile = get_profile_for_chat(chat_id)
        text = build_status_text(m, t, profile)
        nflags, flags = evaluate_flags(m, t)
        lines = [text, ""]
        if nflags > 0:
            lines.append(
                f"⚠️ {b(f'Triggered flags ({nflags})')}: " + ", ".join(flags))
        else:
            lines.append("✅ No flags at current thresholds.")
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")
    finally:
        await dc.close()


async def cmd_assess_json(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id if update and update.effective_chat else None
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        t = get_thresholds_for_chat(chat_id)
        nflags, flags = evaluate_flags(m, t)
        profile = get_profile_for_chat(chat_id)
        certainty, subs = compute_alt_top_certainty(m, t)
        payload = {
            "timestamp": m["timestamp"],
            "chat_id": chat_id,
            "profile": profile,
            "thresholds": t,
            "metrics": m,
            "composite": {"alt_top_certainty": certainty, "subscores": subs},
            "flags_count": nflags,
            "flags": flags,
        }
        import json
        import io
        pretty = True
        as_file = False
        if context.args:
            for a in context.args:
                a = a.lower()
                if a == "file":
                    as_file = True
                elif a == "compact":
                    pretty = False
                elif a == "pretty":
                    pretty = True
        text = json.dumps(payload, indent=(
            2 if pretty else None), ensure_ascii=False)

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


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed. You'll receive alerts and daily summaries here.", parse_mode="HTML")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.discard(update.effective_chat.id)
    await update.message.reply_text("🛑 Unsubscribed.", parse_mode="HTML")


async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args[0]) != 5 or context.args[0][2] != ":":
        await update.message.reply_text("Usage: /settime HH:MM (24h, UTC)", parse_mode="HTML")
        return
    CFG.schedule["daily_summary_time"] = context.args[0]
    await update.message.reply_text(f"🕒 Daily summary time set to {b(context.args[0])} (UTC).", parse_mode="HTML")


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
    cur = get_profile_for_chat(update.effective_chat.id)
    await update.message.reply_text(f"Current risk profile: {b(cur)}", parse_mode="HTML")

# =========================
# Health server & Schedules
# =========================


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


def normalize_tz(tz: str) -> str:
    mapping = {
        "EDT": "America/New_York", "EST": "America/New_York",
        "PDT": "America/Los_Angeles", "PST": "America/Los_Angeles",
        "UTC": "UTC"
    }
    return mapping.get(tz, tz)


def parse_hhmm(hhmm: str) -> Tuple[int, int]:
    hh, mm = hhmm.split(":")
    return int(hh), int(mm)


async def push_summary_for_chat(app: Application, chat_id: int, metrics: Dict[str, Any]):
    t = get_thresholds_for_chat(chat_id)
    profile = get_profile_for_chat(chat_id)
    text = build_status_text(metrics, t, profile)
    n, flags = evaluate_flags(metrics, t)
    if n > 0:
        text += "\n\n" + \
            f"⚠️ {b(f'Triggered flags ({n})')}: " + ", ".join(flags)
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
    except Exception as e:
        log.warning("Failed to send summary to %s: %s", chat_id, e)


async def push_summary(app: Application):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        for chat_id in set(SUBSCRIBERS):
            await push_summary_for_chat(app, chat_id, m)
    finally:
        await dc.close()


async def push_alerts(app: Application):
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        for chat_id in set(SUBSCRIBERS):
            t = get_thresholds_for_chat(chat_id)
            n, flags = evaluate_flags(m, t)
            if n >= t.get("min_flags_for_alert", 3):
                text = "🚨 " + b("Top Risk Alert") + "\n\n" + build_status_text(m, t, get_profile_for_chat(chat_id)) + \
                       "\n\n" + \
                    f"⚠️ {b(f'Triggered flags ({n})')}: " + ", ".join(flags)
                try:
                    await app.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                except Exception as e:
                    log.warning("Failed to send alert to %s: %s", chat_id, e)
    finally:
        await dc.close()

# =========================
# Main
# =========================


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
    await start_health_server()

    tzname = normalize_tz(os.getenv("TZ", "UTC"))
    loop = asyncio.get_running_loop()
    scheduler = AsyncIOScheduler(timezone=tzname, event_loop=loop)

    hh, mm = parse_hhmm(CFG.schedule.get("daily_summary_time", "13:00"))
    scheduler.add_job(push_summary, CronTrigger(
        hour=hh, minute=mm), args=[app])   # daily
    scheduler.add_job(push_alerts, CronTrigger(minute="*/15"),
                      args=[app])         # every 15 min
    scheduler.start()

    if CFG.force_chat_id:
        try:
            await app.bot.send_message(chat_id=int(CFG.force_chat_id), text="🤖 Crypto Cycle Watch bot started.", parse_mode="HTML")
        except Exception as e:
            log.warning("Unable to notify force_chat_id: %s", e)

    log.info("Bot running. Press Ctrl+C to exit.")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
