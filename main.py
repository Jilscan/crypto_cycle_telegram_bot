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
    "schedule": {"daily_summary_time": "13:00"},  # UTC on Koyeb by default
    "symbols": {
        "funding": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
        "oi": ["BTCUSDT", "ETHUSDT"],
    },
    "thresholds": {
        "warn_fraction": 0.80,             # 80% of threshold = 🟡; >=100% = 🔴

        "min_flags_for_alert": 3,

        # Market structure (risk when value is HIGH)
        "btc_dominance_max": 60.0,
        "eth_btc_ratio_max": 0.09,
        "altcap_btc_ratio_max": 1.8,

        # Derivatives (risk when value is HIGH)
        "funding_rate_abs_max": 0.10,  # percent per 8h
        "open_interest_btc_usdt_usd_max": 20_000_000_000,
        "open_interest_eth_usdt_usd_max": 8_000_000_000,

        # Sentiment (risk when value is HIGH)
        "google_trends_7d_avg_min": 75,     # treat as "≥" risk
        "fear_greed_greed_min": 70,         # Greed threshold (0–100)
        "fear_greed_ma14_min": 70,          # 14d average ≥ Greed
        "fear_greed_ma30_min": 65,          # 30d average ≥ Greed (softer)
        "greed_persistence_days": 10,       # greedy streak length
        "greed_persistence_pct30_min": 0.60,  # share of last 30 days in Greed

        # Cycle/On-chain (risk when value is HIGH)
        "mvrv_z_extreme": 7.0,
        "pi_cycle_ratio_min": 0.98,         # 111DMA / (2*350DMA)
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
        }},
        "moderate": {"thresholds": {
            "warn_fraction": 0.80,
            "min_flags_for_alert": 3,
            "btc_dominance_max": 60.0,
            "eth_btc_ratio_max": 0.09,
            "altcap_btc_ratio_max": 1.8,
            "funding_rate_abs_max": 0.10,
            "open_interest_btc_usdt_usd_max": 20_000_000_000,
            "open_interest_eth_usdt_usd_max": 8_000_000_000,
            "google_trends_7d_avg_min": 75,
            "fear_greed_greed_min": 70,
            "fear_greed_ma14_min": 70,
            "fear_greed_ma30_min": 65,
            "greed_persistence_days": 10,
            "greed_persistence_pct30_min": 0.60,
            "mvrv_z_extreme": 7.0,
            "pi_cycle_ratio_min": 0.98,
        }},
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


def get_profile_for_chat(chat_id: Optional[int]) -> str:
    return CHAT_PROFILE.get(chat_id, CFG.default_profile) if CFG.risk_profiles else "static"

# ───────────────────────────
# HTTP helpers & data clients
# ───────────────────────────


OKX_BASE = "https://www.okx.com"
HEADERS = {"User-Agent": "crypto-cycle-bot/1.7", "Accept": "application/json"}


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


def to_okx_uly(sym: str) -> str:
    # "BTCUSDT" -> "BTC-USDT"
    if sym.endswith("USDT"):
        base = sym[:-5]
        quote = "USDT"
    elif sym.endswith("USD"):
        base = sym[:-3]
        quote = "USD"
    else:
        base = sym
        quote = "USDT"
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

    # ---------- Primary: CoinGecko ----------
    async def coingecko_global(self) -> Dict[str, Any]:
        return await fetch_json(self.client, "https://api.coingecko.com/api/v3/global")

    async def coingecko_eth_btc(self) -> float:
        data = await fetch_json(
            self.client,
            "https://api.coingecko.com/api/v3/simple/price",
            {"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
        )
        eth_usd = float(data["ethereum"]["usd"])
        btc_usd = float(data["bitcoin"]["usd"])
        return eth_usd / btc_usd

    async def coingecko_btc_daily(self, days: str | int = "max") -> List[float]:
        data = await fetch_json(
            self.client,
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            {"vs_currency": "usd", "days": days, "interval": "daily"},
        )
        prices = data.get("prices", [])
        return [float(p[1]) for p in prices]

    # ---------- Fallback 1: CoinMarketCap (needs key) ----------
    def _cmc_headers(self) -> Dict[str, str]:
        key = os.getenv("CMC_API_KEY", "")
        return {"X-CMC_PRO_API_KEY": key} if key else {}

    async def cmc_global(self) -> Optional[Dict[str, Any]]:
        if not self._cmc_headers():
            return None
        data = await fetch_json(self.client, "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest", headers=self._cmc_headers())
        return data

    async def cmc_quotes(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        if not self._cmc_headers():
            return None
        syms = ",".join(symbols)
        data = await fetch_json(self.client, "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                                params={"symbol": syms, "convert": "USD"},
                                headers=self._cmc_headers())
        return data

    # ---------- Fallback 2: CoinPaprika ----------
    async def paprika_global(self) -> Optional[Dict[str, Any]]:
        data = await fetch_json(self.client, "https://api.coinpaprika.com/v1/global")
        return data

    # ---------- OKX public (no key) ----------
    async def okx_ticker_last_price(self, instId: str) -> Optional[float]:
        data = await fetch_json(self.client, f"{OKX_BASE}/api/v5/market/ticker", {"instId": instId})
        try:
            lst = data.get("data", [])
            if lst:
                return float(lst[0]["last"])
        except Exception:
            pass
        return None

    async def okx_eth_btc_ratio(self) -> Optional[float]:
        # Spot pair ETH-BTC exists on OKX
        price = await self.okx_ticker_last_price("ETH-BTC")
        return float(price) if price is not None else None

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

    async def okx_open_interest_usd_by_inst(self, instId: str) -> Optional[float]:
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
            if oi_ccy not in (None, "", "0"):
                last = await _safe(self.okx_ticker_last_price(instId), None, f"lastPrice {instId}")
                if last:
                    return float(oi_ccy) * float(last)
        except Exception:
            pass
        return None

    async def okx_instruments_map(self, uly: str) -> Dict[str, Any]:
        data = await fetch_json(
            self.client,
            f"{OKX_BASE}/api/v5/public/instruments",
            {"instType": "SWAP", "uly": uly}
        )
        return {row.get("instId"): row for row in data.get("data", [])}

    async def okx_open_interest_usd_by_uly(self, uly: str) -> Optional[float]:
        oi_data = await fetch_json(
            self.client,
            f"{OKX_BASE}/api/v5/public/open-interest",
            {"instType": "SWAP", "uly": uly}
        )
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

    # ---------- Google Trends ----------
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

    # ---------- Fear & Greed ----------
    async def fear_greed(self) -> Tuple[Optional[int], Optional[str]]:
        try:
            data = await fetch_json(self.client, "https://api.alternative.me/fng/", {"limit": 1})
            row = (data or {}).get("data", [])[0]
            val = int(row["value"])
            label = str(row.get("value_classification") or "")
            return val, label
        except Exception as e:
            log.warning("fear_greed fetch failed: %s", e)
            return None, None

    async def fear_greed_history(self, days: int = 60) -> List[int]:
        try:
            data = await fetch_json(self.client, "https://api.alternative.me/fng/", {"limit": days})
            rows = (data or {}).get("data", [])
            vals = []
            for r in rows:
                try:
                    vals.append(int(r["value"]))
                except Exception:
                    pass
            return vals
        except Exception as e:
            log.warning("fear_greed_history failed: %s", e)
            return []

    # ---------- Optional: Glassnode ----------
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
# Metrics (with redundancy)
# ───────────────────────────


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
    if n % 2:
        return abs(s[mid])
    return abs((s[mid - 1] + s[mid]) / 2.0)


async def _global_with_fallbacks(dc: DataClient) -> Dict[str, Any]:
    # 1) CoinGecko
    cg = await _safe(dc.coingecko_global(), None, "coingecko_global")
    if cg and cg.get("data"):
        return cg

    # 2) CoinMarketCap (requires CMC_API_KEY)
    cmc = await _safe(dc.cmc_global(), None, "cmc_global")
    if cmc and cmc.get("data"):
        d = cmc["data"]
        total = float(d["quote"]["USD"]["total_market_cap"])
        btc_dom = float(d.get("btc_dominance", 0.0))
        # ETH dominance is not always given; try quotes for ETH
        eth_dom = None
        q = await _safe(dc.cmc_quotes(["BTC", "ETH"]), None, "cmc_quotes_btc_eth")
        if q and q.get("data"):
            try:
                btc_mcap = float(q["data"]["BTC"]["quote"]
                                 ["USD"]["market_cap"])
                eth_mcap = float(q["data"]["ETH"]["quote"]
                                 ["USD"]["market_cap"])
                btc_dom = (btc_mcap / total) * 100.0 if total > 0 else btc_dom
                eth_dom = (eth_mcap / total) * 100.0 if total > 0 else None
            except Exception:
                pass
        return {"data": {"total_market_cap": {"usd": total},
                         "market_cap_percentage": {"btc": btc_dom, "eth": eth_dom or 0.0}}}

    # 3) CoinPaprika
    pk = await _safe(dc.paprika_global(), None, "paprika_global")
    if pk and isinstance(pk, dict):
        total = float(pk.get("market_cap_usd", 0.0))
        btc_dom = float(pk.get("bitcoin_dominance_percentage", 0.0))
        # No ETH dominance; leave 0.0 so altcap will be slightly overstated in rare fallback
        return {"data": {"total_market_cap": {"usd": total},
                         "market_cap_percentage": {"btc": btc_dom, "eth": 0.0}}}

    # Last resort: zeros
    return {"data": {"total_market_cap": {"usd": 0.0},
                     "market_cap_percentage": {"btc": 0.0, "eth": 0.0}}}


async def _eth_btc_with_fallback(dc: DataClient) -> float:
    v = await _safe(dc.coingecko_eth_btc(), None, "coingecko_eth_btc")
    if v is not None and v > 0:
        return float(v)
    okx = await _safe(dc.okx_eth_btc_ratio(), None, "okx_eth_btc_ratio")
    return float(okx or 0.0)


async def gather_metrics(dc: DataClient) -> Dict[str, Any]:
    cg = await _global_with_fallbacks(dc)
    ethbtc = await _eth_btc_with_fallback(dc)

    # funding across basket (OKX)
    funding: Dict[str, Dict[str, Any]] = {}
    abs_funding_list: List[float] = []
    for sym in CFG.symbols["funding"]:
        inst = to_okx_instId(sym)
        fr = await _safe(dc.okx_funding_rate(inst), 0.0, f"funding {inst}")
        lp = await _safe(dc.okx_ticker_last_price(inst), 0.0, f"lastPrice {inst}")
        funding[sym] = {"lastFundingRate": fr or 0.0, "markPrice": lp or 0.0}
        abs_funding_list.append(abs((fr or 0.0) * 100.0))

    # funding stats (extreme + median + top3)
    max_fr = 0.0
    max_sym = None
    for sym, d in funding.items():
        fr_pct = abs(float(d.get("lastFundingRate", 0.0)) * 100.0)
        if fr_pct > max_fr:
            max_fr = fr_pct
            max_sym = sym
    median_fr = _median_abs(abs_funding_list) or 0.0
    top3 = sorted(((sym, abs(d.get("lastFundingRate", 0.0)) * 100.0) for sym, d in funding.items()),
                  key=lambda x: x[1], reverse=True)[:3]

    # open interest USD (OKX) robust
    oi_usd: Dict[str, Optional[float]] = {}
    for sym in CFG.symbols["oi"]:
        inst = to_okx_instId(sym)
        uly = to_okx_uly(sym)
        notional = await _safe(dc.okx_open_interest_usd_by_inst(inst), None, f"openInterestUSD {inst}")
        if notional is None:
            notional = await _safe(dc.okx_open_interest_usd_by_uly(uly), None, f"openInterestUSD {uly}")
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

    # Fear & Greed persistence and moving averages
    def mean(arr: List[float]) -> Optional[float]:
        return (sum(arr) / len(arr)) if arr else None

    # newest first
    fng_hist = await _safe(dc.fear_greed_history(60), [], "fng_history")
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

    mvrv_z = await _safe(dc.glassnode_mvrv_z("BTC"), None, "glassnode_mvrv_z")
    exch_inflow = await _safe(dc.glassnode_exchange_inflow("BTC"), None, "glassnode_exchange_inflow")

    # Pi Cycle Top proximity (robust CG fetch)
    btc_daily = await _safe(dc.coingecko_btc_daily("max"), [], "btc_daily_prices")
    if len(btc_daily) < 350:
        btc_daily = await _safe(dc.coingecko_btc_daily(1500), btc_daily, "btc_daily_1500")
    if len(btc_daily) < 350:
        btc_daily = await _safe(dc.coingecko_btc_daily(500), btc_daily, "btc_daily_500")

    def _sma(vals: List[float], window: int) -> Optional[float]:
        if len(vals) < window:
            return None
        return sum(vals[-window:]) / float(window)

    sma111 = _sma(btc_daily, 111)
    sma350 = _sma(btc_daily, 350)
    pi_ratio = None
    two_350 = None
    if sma111 is not None and sma350 is not None and sma350 > 0:
        two_350 = 2.0 * sma350
        pi_ratio = sma111 / two_350

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
            "top3_abs_pct": top3,  # list of (sym, abs_pct)
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
    }

# ───────────────────────────
# Composite score (Alt-Top Certainty 0–100)
# ───────────────────────────


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def compute_alt_top_certainty(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, Dict[str, float]]:
    subscores: Dict[str, float] = {}

    def frac(value: Optional[float], threshold: Optional[float]) -> float:
        if value is None or threshold is None or threshold <= 0:
            return 0.0
        return clamp01(float(value) / float(threshold))

    # Market structure
    subscores["altcap_vs_btc"] = frac(
        m["altcap_btc_ratio"], t["altcap_btc_ratio_max"])
    subscores["eth_btc"] = frac(m["eth_btc"],           t["eth_btc_ratio_max"])
    subscores["btc_dominance"] = frac(
        m["btc_dominance_pct"], t["btc_dominance_max"])

    # Derivatives
    max_fr_abs = float(m.get("funding_stats", {}).get("max_abs_pct", 0.0))
    subscores["funding_extreme"] = frac(max_fr_abs, t["funding_rate_abs_max"])
    oi = m.get("open_interest_usd") or {}
    subscores["oi_btc"] = frac(
        oi.get("BTCUSDT"), t["open_interest_btc_usdt_usd_max"])
    subscores["oi_eth"] = frac(
        oi.get("ETHUSDT"), t["open_interest_eth_usdt_usd_max"])

    # Sentiment
    subscores["trends"] = frac(
        m["google_trends_avg7d"], t["google_trends_7d_avg_min"])
    fg_val = m.get("fear_greed_index")
    subscores["fear_greed"] = frac(float(fg_val) if fg_val is not None else None,
                                   float(t.get("fear_greed_greed_min", 70)))

    # F&G moving averages & persistence
    ma14 = m.get("fear_greed_ma14")
    ma30 = m.get("fear_greed_ma30")
    streak = m.get("fear_greed_greed_streak_days")
    pct30 = m.get("fear_greed_pct30d_greed")

    subscores["fng_ma14"] = frac(ma14, t.get("fear_greed_ma14_min", 70))
    subscores["fng_ma30"] = frac(ma30, t.get("fear_greed_ma30_min", 65))
    p_days = frac(streak, t.get("greed_persistence_days", 10))
    p_pct = frac(pct30, t.get("greed_persistence_pct30_min", 0.60))
    subscores["fng_persist"] = max(p_days, p_pct)

    # Cycle/On-chain
    pi_ratio = (m.get("pi_cycle") or {}).get("ratio")
    subscores["pi_cycle"] = frac(pi_ratio, t.get("pi_cycle_ratio_min", 0.98))

    mz = m.get("mvrv_z")
    subscores["mvrv_z"] = frac(
        mz, float(t.get("mvrv_z_extreme", 7.0))) if mz is not None else 0.0

    # Weights (renormalized automatically)
    weights = {
        # structure
        "altcap_vs_btc": 0.16,
        "eth_btc":       0.13,
        "btc_dominance": 0.07,
        # derivatives
        "funding_extreme": 0.12,
        "oi_btc":        0.11,
        "oi_eth":        0.09,
        # sentiment
        "trends":        0.06,
        "fear_greed":    0.06,
        "fng_ma14":      0.06,
        "fng_ma30":      0.04,
        "fng_persist":   0.06,
        # cycle/on-chain
        "pi_cycle":      0.06,
        "mvrv_z":        0.04,
    }

    total_w = sum(weights.values())
    if total_w <= 0:
        return 0, subscores

    score01 = 0.0
    for k, w in weights.items():
        score01 += subscores.get(k, 0.0) * w
    score01 /= total_w

    return int(round(score01 * 100)), subscores

# ───────────────────────────
# Presentation helpers (HTML) with 3-color severities
# ───────────────────────────


def severity_above(value: Optional[float], thr: Optional[float], warn_frac: float) -> str:
    """Return 'red' if value >= thr, 'yellow' if >= warn_frac*thr, else 'green'. None -> 'green'."""
    if value is None or thr is None or thr <= 0:
        return "green"
    v = float(value)
    if v >= float(thr):
        return "red"
    if v >= float(thr) * float(warn_frac):
        return "yellow"
    return "green"


def bullet3(sev: str) -> str:
    return "🔴" if sev == "red" else "🟡" if sev == "yellow" else "🟢"


def b(txt: str) -> str:
    return f"<b>{txt}</b>"


def fmt_usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def build_status_text(m: Dict[str, Any], t: Dict[str, Any] | None = None, profile_name: str = "") -> str:
    t = t or CFG.thresholds
    warn_frac = float(t.get("warn_fraction", 0.8))

    # MARKET STRUCTURE
    dom = m["btc_dominance_pct"]
    sev_dom = severity_above(dom, t["btc_dominance_max"], warn_frac)

    ethbtc = m["eth_btc"]
    sev_ethbtc = severity_above(ethbtc, t["eth_btc_ratio_max"], warn_frac)

    altbtc = m["altcap_btc_ratio"]
    sev_altbtc = severity_above(altbtc, t["altcap_btc_ratio_max"], warn_frac)

    # DERIVATIVES
    fstats = m.get("funding_stats", {})
    max_fr = float(fstats.get("max_abs_pct", 0.0))
    sev_fund = severity_above(max_fr, t["funding_rate_abs_max"], warn_frac)

    median_fr = float(fstats.get("median_abs_pct", 0.0))
    sev_fund_median = severity_above(
        median_fr, t["funding_rate_abs_max"], warn_frac)

    top3 = fstats.get("top3_abs_pct", []) or []
    oi = m["open_interest_usd"]
    btc_oi = oi.get("BTCUSDT")
    eth_oi = oi.get("ETHUSDT")
    sev_btc_oi = severity_above(
        btc_oi, t["open_interest_btc_usdt_usd_max"], warn_frac)
    sev_eth_oi = severity_above(
        eth_oi, t["open_interest_eth_usdt_usd_max"], warn_frac)

    # SENTIMENT
    gavg = m["google_trends_avg7d"]
    sev_trends = severity_above(gavg, t["google_trends_7d_avg_min"], warn_frac)
    fng_val = m.get("fear_greed_index")
    sev_fng_now = severity_above(float(fng_val) if fng_val is not None else None, t.get(
        "fear_greed_greed_min", 70), warn_frac)

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

    # CYCLE / ON-CHAIN
    pi = m.get("pi_cycle") or {}
    pi_ratio = pi.get("ratio")
    sev_pi = severity_above(pi_ratio, t.get(
        "pi_cycle_ratio_min", 0.98), warn_frac)
    mz_val = m.get("mvrv_z")
    sev_mz = severity_above(mz_val, t.get("mvrv_z_extreme", 7.0), warn_frac)

    # Composite certainty
    certainty, subs = compute_alt_top_certainty(m, t)

    lines: List[str] = []
    lines.append(f"📊 {b('Crypto Market Snapshot')} — {m['timestamp']} UTC")
    if profile_name:
        lines.append(f"Profile: {b(profile_name)}")
    lines.append("")

    # MARKET STRUCTURE
    lines.append(b("Market Structure"))
    lines.append(f"• Bitcoin market share of total crypto: {bullet3(sev_dom)} {b(f'{dom:.2f}%')}  "
                 f"(warn ≥ {t['warn_fraction']*t['btc_dominance_max']:.2f}%, flag ≥ {t['btc_dominance_max']:.2f}%)")
    lines.append(f"• Ether price relative to Bitcoin (ETH/BTC): {bullet3(sev_ethbtc)} {b(f'{ethbtc:.5f}')}  "
                 f"(warn ≥ {t['warn_fraction']*t['eth_btc_ratio_max']:.5f}, flag ≥ {t['eth_btc_ratio_max']:.5f})")
    lines.append(f"• Altcoin market cap / Bitcoin market cap: {bullet3(sev_altbtc)} {b(f'{altbtc:.2f}')}  "
                 f"(warn ≥ {t['warn_fraction']*t['altcap_btc_ratio_max']:.2f}, flag ≥ {t['altcap_btc_ratio_max']:.2f})")
    lines.append("")

    # DERIVATIVES
    lines.append(b("Derivatives"))
    # basket detail
    sym_list = ", ".join(CFG.symbols.get("funding", []))
    if top3:
        t3 = ", ".join([f"{s} {v:.3f}%" for s, v in top3])
    else:
        t3 = "n/a"
    lines.append(f"• Funding (basket: {sym_list}) — max: {bullet3(sev_fund)} {b(f'{max_fr:.3f}%')} | "
                 f"median: {bullet3(sev_fund_median)} {b(f'{median_fr:.3f}%')}  "
                 f"(warn ≥ {t['warn_fraction']*t['funding_rate_abs_max']:.3f}%, flag ≥ {t['funding_rate_abs_max']:.3f}%)")
    lines.append(f"  Top-3 funding extremes: {b(t3)}")
    lines.append(f"• Bitcoin open interest (USD): {bullet3(sev_btc_oi)} {b(fmt_usd(btc_oi))}  "
                 f"(warn ≥ {fmt_usd(t['warn_fraction']*t['open_interest_btc_usdt_usd_max'])}, "
                 f"flag ≥ {fmt_usd(t['open_interest_btc_usdt_usd_max'])})")
    lines.append(f"• Ether open interest (USD): {bullet3(sev_eth_oi)} {b(fmt_usd(eth_oi))}  "
                 f"(warn ≥ {fmt_usd(t['warn_fraction']*t['open_interest_eth_usdt_usd_max'])}, "
                 f"flag ≥ {fmt_usd(t['open_interest_eth_usdt_usd_max'])})")
    lines.append("")

    # SENTIMENT
    lines.append(b("Sentiment"))
    lines.append(f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {bullet3(sev_trends)} {b(f'{gavg:.1f}')}  "
                 f"(warn ≥ {t['warn_fraction']*t['google_trends_7d_avg_min']:.1f}, flag ≥ {t['google_trends_7d_avg_min']:.1f})")
    if fng_val is not None:
        lines.append(f"• Fear & Greed Index (overall crypto): {bullet3(sev_fng_now)} {b(str(fng_val))} "
                     f"({m.get('fear_greed_label') or 'n/a'})  "
                     f"(warn ≥ {int(t['warn_fraction']*t.get('fear_greed_greed_min',70))}, "
                     f"flag ≥ {t.get('fear_greed_greed_min',70)})")
    else:
        lines.append("• Fear & Greed Index: 🟢 " + b("n/a"))
    lines.append(f"• Fear & Greed 14-day average: {bullet3(sev_ma14)} "
                 f"{b('n/a' if fng_ma14 is None else f'{fng_ma14:.1f}')}  "
                 f"(warn ≥ {int(t['warn_fraction']*t.get('fear_greed_ma14_min',70))}, "
                 f"flag ≥ {t.get('fear_greed_ma14_min',70)})")
    lines.append(f"• Fear & Greed 30-day average: {bullet3(sev_ma30)} "
                 f"{b('n/a' if fng_ma30 is None else f'{fng_ma30:.1f}')}  "
                 f"(warn ≥ {int(t['warn_fraction']*t.get('fear_greed_ma30_min',65))}, "
                 f"flag ≥ {t.get('fear_greed_ma30_min',65)})")
    # persistence shows both streak and pct30; show worst severity of the two
    sev_persist = "red" if (sev_streak == "red" or sev_pct30 == "red") else (
        "yellow" if (sev_streak == "yellow" or sev_pct30 == "yellow") else "green")
    lines.append(f"• Greed persistence: {bullet3(sev_persist)} "
                 f"{b(f'{streak} days in a row')} | "
                 f"{b('n/a' if pct30 is None else f'{pct30*100:.0f}% of last 30 days ≥ {int(t.get(\"fear_greed_greed_min\",70))}')}  "
                 f"(warn: days ≥ {int(t['warn_fraction']*t.get('greed_persistence_days',10))} "
                 f"or pct ≥ {int(100*t['warn_fraction']*t.get('greed_persistence_pct30_min',0.60))}% | "
                 f"flag: days ≥ {t.get('greed_persistence_days',10)} "
                 f"or pct ≥ {int(100*t.get('greed_persistence_pct30_min',0.60))}%)")
    lines.append("")

    # CYCLE / ON-CHAIN
    lines.append(b("Cycle & On-Chain"))
    if pi_ratio is not None and pi.get("sma111") and pi.get("sma350x2"):
        pct = pi_ratio * 100.0
        th_pct = t.get("pi_cycle_ratio_min", 0.98) * 100.0
        lines.append(
            f"• Pi Cycle Top proximity (111-DMA vs 2×350-DMA): {bullet3(sev_pi)} {b(f'{pct:.1f}% of cross')}  "
            f"(111DMA {fmt_usd(pi['sma111'])}, 2×350DMA {fmt_usd(pi['sma350x2'])}; "
            f"warn ≥ {t['warn_fraction']*th_pct:.1f}%, flag ≥ {th_pct:.1f}%)"
        )
    else:
        lines.append("• Pi Cycle Top proximity: 🟢 " + b("n/a"))
    if mz_val is not None:
        lines.append(f"• Bitcoin MVRV Z-Score: {bullet3(sev_mz)} {b(f'{mz_val:.2f}')}  "
                     f"(warn ≥ {t['warn_fraction']*t['mvrv_z_extreme']:.2f}, flag ≥ {t['mvrv_z_extreme']:.2f})")
    lines.append("")

    # COMPOSITE
    certainty, subs = compute_alt_top_certainty(m, t)
    lines.append(b("Alt-Top Certainty (Composite)"))
    lines.append(f"• Certainty: {b(f'{certainty}/100')}")
    nice = [
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
        f"MVRVZ {int(round(subs.get('mvrv_z',0)*100))}%"
    ]
    lines.append("• Subscores: " + ", ".join(nice))

    return "\n".join(lines)

# ───────────────────────────
# Flags (same logic as before)
# ───────────────────────────


def evaluate_flags(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[str]]:
    flags: List[str] = []
    if m["btc_dominance_pct"] >= t["btc_dominance_max"]:
        flags.append("High Bitcoin dominance")
    if m["eth_btc"] >= t["eth_btc_ratio_max"]:
        flags.append("Elevated ETH/BTC ratio")
    if m["altcap_btc_ratio"] >= t["altcap_btc_ratio_max"]:
        flags.append("Alt market cap stretched vs BTC")
    # funding (use basket max)
    max_fr = float(m.get("funding_stats", {}).get("max_abs_pct", 0.0))
    if max_fr >= t["funding_rate_abs_max"]:
        # include the top-coin name when available
        max_sym = m.get("funding_stats", {}).get("top3_abs_pct", [])
        if max_sym:
            flags.append(
                f"Perpetual funding extreme ({max_sym[0][0]} {max_fr:.3f}%)")
        else:
            flags.append(f"Perpetual funding extreme ({max_fr:.3f}%)")
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
    # F&G moving averages and persistence
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
    # Pi Cycle, MVRV
    pi = m.get("pi_cycle") or {}
    if pi.get("ratio") is not None and pi["ratio"] >= t.get("pi_cycle_ratio_min", 0.98):
        flags.append("Pi Cycle Top proximity elevated")
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
        "👋 Crypto Cycle Watch online.\n\n"
        "Commands:\n"
        "• /status – snapshot\n"
        "• /assess – snapshot + flags\n"
        "• /assess_json [pretty|compact|file] – raw data\n"
        "• /subscribe – daily summary & alerts here\n"
        "• /unsubscribe – stop messages\n"
        "• /risk &lt;conservative|moderate|aggressive&gt;\n"
        "• /getrisk – show profile\n"
        "• /settime HH:MM – daily summary time (server/UTC)\n"
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


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("✅ Subscribed. You'll receive alerts and daily summaries here.", parse_mode="HTML")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.discard(update.effective_chat.id)
    await update.message.reply_text("🛑 Unsubscribed.", parse_mode="HTML")


async def cmd_settime(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args[0]) != 5 or context.args[0][2] != ":":
        await update.message.reply_text("Usage: /settime HH:MM (24h, server/UTC)", parse_mode="HTML")
        return
    CFG.schedule["daily_summary_time"] = context.args[0]
    await update.message.reply_text(f"🕒 Daily summary time set to {b(context.args[0])} (server/UTC).", parse_mode="HTML")


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


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
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


def normalize_tz(tz: str) -> str:
    mapping = {
        "EDT": "America/New_York", "EST": "America/New_York",
        "PDT": "America/Los_Angeles", "PST": "America/Los_Angeles",
        "UTC": "UTC"
    }
    return mapping.get(tz, tz)

# ───────────────────────────
# Alerts / Summaries
# ───────────────────────────


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

    # Health server first so health checks pass
    await start_health_server()

    # Scheduler — daily summary & 15-min alerts
    tzname = normalize_tz(os.getenv("TZ", "UTC"))
    loop = asyncio.get_running_loop()
    scheduler = AsyncIOScheduler(timezone=tzname, event_loop=loop)

    hh, mm = parse_hhmm(CFG.schedule.get("daily_summary_time", "13:00"))
    scheduler.add_job(push_summary, CronTrigger(
        hour=hh, minute=mm), args=[app])   # daily summary
    scheduler.add_job(push_alerts, CronTrigger(
        minute="*/15"), args=[app])         # alert check
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
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
