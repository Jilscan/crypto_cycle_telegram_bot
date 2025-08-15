#!/usr/bin/env python3
import os
import asyncio
import signal
import math
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Telegram
from telegram.ext import (
    Application, CommandHandler, ContextTypes
)
from telegram import Update

# Google Trends
from pytrends.request import TrendReq

# Data wrangling
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("crypto-cycle-bot")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def pretty_pct(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    return f"{x:.3f}%"


def pretty_usd(x: Optional[float]) -> str:
    if x is None or math.isnan(x):
        return "n/a"
    if x >= 1_000_000_000:
        return f"${x/1_000_000_000:.3f}B"
    if x >= 1_000_000:
        return f"${x/1_000_000:.3f}M"
    return f"${x:,.0f}"


def flag_color(value: float, warn: float, flag: float, direction: str = "above") -> str:
    """
    direction = 'above' means risk increases as value goes up (>= warn / flag)
    direction = 'below' means risk increases as value goes down (<= warn / flag)
    returns "🟢"/"🟡"/"🔴"
    """
    if math.isnan(value):
        return "🟡"
    if direction == "above":
        if value >= flag:
            return "🔴"
        elif value >= warn:
            return "🟡"
        else:
            return "🟢"
    else:
        # risk if too low
        if value <= flag:
            return "🔴"
        elif value <= warn:
            return "🟡"
        else:
            return "🟢"


def traffic_from_score(score: int) -> str:
    if score >= 70:
        return "🔴"
    if score >= 40:
        return "🟡"
    return "🟢"


def split_long_message(txt: str, limit: int = 3800) -> List[str]:
    lines = txt.splitlines(True)
    chunks: List[str] = []
    cur = ""
    for ln in lines:
        if len(cur) + len(ln) > limit:
            chunks.append(cur)
            cur = ln
        else:
            cur += ln
    if cur:
        chunks.append(cur)
    return chunks

# --------------------------------------------------------------------------------------
# Indicators (RSI / Stoch RSI / SMA)
# --------------------------------------------------------------------------------------


def rsi_from_closes(closes: List[float], period: int = 14) -> List[float]:
    arr = pd.Series(closes, dtype=float)
    delta = arr.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").tolist()


def sma(values: List[float], period: int) -> List[float]:
    s = pd.Series(values, dtype=float)
    return s.rolling(period).mean().fillna(method="bfill").tolist()


def stoch_rsi_kd(closes: List[float], rsi_period: int = 14, stoch_period: int = 14,
                 smooth_k: int = 3, smooth_d: int = 3) -> Tuple[List[float], List[float]]:
    r = pd.Series(rsi_from_closes(closes, rsi_period), dtype=float)
    low = r.rolling(stoch_period).min()
    high = r.rolling(stoch_period).max()
    stoch = (r - low) / (high - low)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    k = k.fillna(method="bfill").clip(lower=0, upper=1)
    d = d.fillna(method="bfill").clip(lower=0, upper=1)
    return k.tolist(), d.tolist()


def to_2w_series(ts: List[int], closes: List[float]) -> Tuple[List[int], List[float]]:
    """Collapse weekly series into 2-week series by pairing sequential bars."""
    if len(closes) < 3:
        return ts, closes
    out_ts, out_cl = [], []
    i = 0
    while i < len(closes):
        j = min(i + 1, len(closes) - 1)
        out_ts.append(ts[j])
        out_cl.append(closes[j])
        i += 2
    return out_ts, out_cl

# --------------------------------------------------------------------------------------
# Data client
# --------------------------------------------------------------------------------------


class DataClient:
    def __init__(self):
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, read=20.0, connect=15.0))

    async def close(self):
        await self.http.aclose()

    # ------------- Coingecko (lightweight only) -------------
    async def coingecko_global(self) -> Optional[Dict[str, Any]]:
        try:
            r = await self.http.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            return r.json().get("data", {})
        except Exception as e:
            log.warning("coingecko_global failed: %s", e)
            return None

    async def coingecko_prices(self) -> Dict[str, float]:
        out = {"bitcoin": float("nan"), "ethereum": float("nan")}
        try:
            r = await self.http.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": "bitcoin,ethereum", "vs_currencies": "usd"},
            )
            r.raise_for_status()
            js = r.json()
            out["bitcoin"] = safe_float(
                js.get("bitcoin", {}).get("usd"), float("nan"))
            out["ethereum"] = safe_float(
                js.get("ethereum", {}).get("usd"), float("nan"))
        except Exception as e:
            log.warning("coingecko_prices failed: %s", e)
        return out

    # ------------- OKX -------------
    async def okx_get(self, path: str, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        try:
            r = await self.http.get(f"https://www.okx.com{path}", params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("OKX GET %s failed: %s", path, e)
            return None

    async def okx_funding_and_last(self, inst_id: str) -> Tuple[Optional[float], Optional[float]]:
        fr = await self.okx_get("/api/v5/public/funding-rate", {"instId": inst_id})
        last = await self.okx_get("/api/v5/market/ticker", {"instId": inst_id})
        rate = None
        price = None
        try:
            if fr and fr.get("data"):
                rate = 100 * \
                    safe_float(fr["data"][0].get("fundingRate"), float("nan"))
        except Exception:
            pass
        try:
            if last and last.get("data"):
                price = safe_float(last["data"][0].get("last"), float("nan"))
        except Exception:
            pass
        return rate, price

    async def okx_open_interest_usd(self, inst_id: str) -> Optional[float]:
        oi_js = await self.okx_get("/api/v5/public/open-interest", {"instType": "SWAP", "instId": inst_id})
        last_js = await self.okx_get("/api/v5/market/ticker", {"instId": inst_id})
        try:
            if not oi_js or not oi_js.get("data"):
                return None
            row = oi_js["data"][0]
            # open interest in coin units
            oi_ccy = safe_float(row.get("oiCcy"), float("nan"))
            if math.isnan(oi_ccy):
                # fallback to contracts (rough)
                oi = safe_float(row.get("oi"), float("nan"))
                if math.isnan(oi):
                    return None
                # Try to multiply by price directly
                price = None
                if last_js and last_js.get("data"):
                    price = safe_float(
                        last_js["data"][0].get("last"), float("nan"))
                if price and not math.isnan(price):
                    return oi * price
                return None
            price = None
            if last_js and last_js.get("data"):
                price = safe_float(
                    last_js["data"][0].get("last"), float("nan"))
            if price and not math.isnan(price):
                return oi_ccy * price
        except Exception as e:
            log.warning("okx_open_interest_usd failed: %s", e)
        return None

    async def okx_candles(self, inst_id: str, bar: str, limit: int) -> Tuple[List[int], List[float]]:
        """
        Returns (timestamps_ms asc, closes asc)
        OKX returns newest-first; we’ll reverse.
        """
        js = await self.okx_get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
        if not js or not js.get("data"):
            return [], []
        rows = js["data"]
        rows.reverse()
        ts = [int(r[0]) for r in rows]
        closes = [safe_float(r[4], float("nan")) for r in rows]
        return ts, closes

    # ------------- Google Trends -------------
    async def google_trends_score(self) -> Optional[float]:
        try:
            # pytrends is sync; run in a thread so we don’t block.
            loop = asyncio.get_running_loop()

            def run():
                py = TrendReq(hl="en-US", tz=0)
                py.build_payload(
                    ["crypto", "bitcoin", "ethereum"], timeframe="today 3-m")
                df = py.interest_over_time()
                if df is None or df.empty:
                    return None
                # last 7 days mean across three columns
                tail = df.tail(7)[["crypto", "bitcoin", "ethereum"]]
                return float(tail.mean().mean())
            return await loop.run_in_executor(None, run)
        except Exception as e:
            log.warning("google_trends_score failed: %s", e)
            return None

    # ------------- Fear & Greed -------------
    async def fear_greed(self) -> Tuple[Optional[int], Optional[float], Optional[float], int, float]:
        """
        Returns: (today, ma14, ma30, greed_run_days, percent_greed30)
        greed threshold: >=70
        """
        today = None
        ma14 = None
        ma30 = None
        greed_run_days = 0
        greed_pct30 = float("nan")
        try:
            r1 = await self.http.get("https://api.alternative.me/fng/?limit=1")
            r1.raise_for_status()
            j1 = r1.json()
            if j1.get("data"):
                today = int(j1["data"][0]["value"])

            r60 = await self.http.get("https://api.alternative.me/fng/?limit=60")
            r60.raise_for_status()
            j60 = r60.json()
            vals = [int(x["value"])
                    for x in j60.get("data", [])][::-1]  # oldest->newest
            if vals:
                s = pd.Series(vals, dtype=float)
                if len(s) >= 14:
                    ma14 = float(s.rolling(14).mean().iloc[-1])
                if len(s) >= 30:
                    ma30 = float(s.rolling(30).mean().iloc[-1])
                    last30 = s.iloc[-30:]
                    greed_pct30 = float((last30 >= 70).mean() * 100.0)

                # consecutive greed run (from end)
                i = len(s) - 1
                while i >= 0 and s.iloc[i] >= 70:
                    greed_run_days += 1
                    i -= 1
        except Exception as e:
            log.warning("fear_greed failed: %s", e)
        return today, ma14, ma30, greed_run_days, greed_pct30

    # ------------- CMC Total3 (optional) -------------
    async def cmc_total3(self) -> Optional[float]:
        """
        Returns Total3 market cap in USD if CMC_API_KEY present.
        """
        api_key = os.getenv("CMC_API_KEY")
        if not api_key:
            return None
        try:
            headers = {"X-CMC_PRO_API_KEY": api_key}
            # Global metrics with market cap details
            url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
            r = await self.http.get(url, headers=headers, params={})
            r.raise_for_status()
            data = r.json().get("data", {})
            # CMC doesn't directly return Total3; approximate: TOTAL_MARKET_CAP - BTC_CAP - ETH_CAP
            total = safe_float(data.get("quote", {}).get(
                "USD", {}).get("total_market_cap"), float("nan"))
            btc_dominance = safe_float(
                data.get("btc_dominance"), float("nan"))  # %
            eth_dominance = safe_float(
                data.get("eth_dominance"), float("nan"))  # %
            if not math.isnan(total) and not math.isnan(btc_dominance) and not math.isnan(eth_dominance):
                btc_cap = total * (btc_dominance / 100.0)
                eth_cap = total * (eth_dominance / 100.0)
                total3 = total - btc_cap - eth_cap
                return total3
        except Exception as e:
            log.warning("cmc_total3 failed: %s", e)
        return None


# --------------------------------------------------------------------------------------
# Metrics computation
# --------------------------------------------------------------------------------------
ALT_BASKET = ["SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
              "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"]
FUNDING_INSTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP",
                 "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]


def profile_thresholds(profile: str) -> Dict[str, Any]:
    # These thresholds tune how sensitive the flags are
    p = profile.lower().strip()
    if p not in {"conservative", "moderate", "aggressive"}:
        p = "moderate"
    scale = {"conservative": 0.9, "moderate": 1.0, "aggressive": 1.1}[p]
    # interpret scale: conservative warns earlier (lower warn), aggressive warns later (higher warn)
    t = {
        "btc_dom_warn": 48.0 * (1/scale),
        "btc_dom_flag": 60.0 * (1/scale),
        "ethbtc_warn": 0.072 * scale,
        "ethbtc_flag": 0.090 * scale,
        "alt_btc_warn": 1.44 * scale,
        "alt_btc_flag": 1.80 * scale,
        "funding_warn": 0.08 * scale * 100,  # percent
        "funding_flag": 0.10 * scale * 100,
        "oi_btc_warn": 16e9 * scale,
        "oi_btc_flag": 20e9 * scale,
        "oi_eth_warn": 6.4e9 * scale,
        "oi_eth_flag": 8e9 * scale,
        "trends_warn": 60.0 * scale,
        "trends_flag": 75.0 * scale,
        "fng_warn": 56 * scale,
        "fng_flag": 70 * scale,
        "fng14_warn": 56 * scale,
        "fng14_flag": 70 * scale,
        "fng30_warn": 52 * scale,
        "fng30_flag": 65 * scale,
        "fng_days_warn": int(round(8 / scale)),
        "fng_days_flag": int(round(10 / scale)),
        "fng_pct30_warn": 48.0 / scale,
        "fng_pct30_flag": 60.0 / scale,
        "rsi_btc2w_warn": 60.0 * scale,
        "rsi_btc2w_flag": 70.0 * scale,
        "rsi_ethbtc2w_warn": 55.0 * scale,
        "rsi_ethbtc2w_flag": 65.0 * scale,
        "rsi_alt2w_warn": 65.0 * scale,
        "rsi_alt2w_flag": 75.0 * scale,
        "stoch_ob": 0.80,  # overbought threshold for Stoch RSI
        "fib_warn_pct": 3.0 / scale,
        "fib_flag_pct": 1.5 / scale,
        "pi_close_pct": 3.0,  # within 3% is very close
    }
    return t


async def gather_metrics(dc: DataClient, prof: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    t = profile_thresholds(prof)

    # --- Market structure ---
    cg = await dc.coingecko_global()
    prices = await dc.coingecko_prices()
    btc_usd = prices.get("bitcoin", float("nan"))
    eth_usd = prices.get("ethereum", float("nan"))
    eth_btc = (eth_usd / btc_usd) if (btc_usd and btc_usd >
                                      0 and not math.isnan(btc_usd) and not math.isnan(eth_usd)) else float("nan")

    btc_dom = float("nan")
    alt_btc_ratio = float("nan")
    if cg:
        try:
            btc_dom = safe_float(
                cg.get("market_cap_percentage", {}).get("btc"), float("nan"))
            if not math.isnan(btc_dom):
                # Alt/BTC ratio = (100 - btc%) / btc%
                alt_btc_ratio = (100.0 - btc_dom) / btc_dom
        except Exception:
            pass

    # If CMC key present, we *could* refine alt_btc_ratio using Total3 / BTC cap, but we lack BTC cap from CMC here.
    # Keeping the Coingecko-derived ratio which worked reliably for you.

    # --- Funding basket + OI from OKX ---
    funding_rows: List[Tuple[str, Optional[float]]] = []
    funding_values: List[float] = []
    for inst in FUNDING_INSTS:
        rate, _ = await dc.okx_funding_and_last(inst)
        funding_rows.append((inst.replace("-USDT-SWAP", "USDT"), rate))
        if rate is not None and not math.isnan(rate):
            funding_values.append(rate)

    funding_max = max(funding_values) if funding_values else float("nan")
    funding_median = float(np.median(funding_values)
                           ) if funding_values else float("nan")

    oi_btc_usd = await dc.okx_open_interest_usd("BTC-USDT-SWAP")
    oi_eth_usd = await dc.okx_open_interest_usd("ETH-USDT-SWAP")

    # --- Google Trends ---
    trends = await dc.google_trends_score()

    # --- Fear & Greed ---
    fng_today, fng_ma14, fng_ma30, fng_run_days, fng_pct30 = await dc.fear_greed()

    # --- Pi Cycle Top proximity (OKX daily) ---
    ts_d, cl_d = await dc.okx_candles("BTC-USDT", "1D", 500)
    # percent distance between 111DMA and 2x350DMA (lower = closer)
    pi_prox = None
    if len(cl_d) >= 350:
        c = pd.Series(cl_d, dtype=float)
        sma111 = c.rolling(111).mean()
        sma350 = c.rolling(350).mean()
        v111 = float(sma111.iloc[-1])
        v2x350 = float(2.0 * sma350.iloc[-1])
        if v111 > 0 and v2x350 > 0:
            dist_pct = abs(v111 - v2x350) / v2x350 * 100.0
            pi_prox = dist_pct

    # --- Weekly data for RSI/Stoch/Fibs ---
    # BTC 1W (and later collapsed to 2W)
    ts_w_btc, cl_w_btc = await dc.okx_candles("BTC-USDT", "1W", 400)
    ts_w_ethbtc, cl_w_ethbtc = await dc.okx_candles("ETH-BTC", "1W", 400)

    # ALT basket 1W
    alt_closes_map: Dict[str, List[float]] = {}
    for inst in ALT_BASKET:
        _, cl = await dc.okx_candles(inst, "1W", 400)
        if cl:
            alt_closes_map[inst] = cl

    # 2-Week series (close-only)
    def to_2w(closes: List[float]) -> List[float]:
        _, cc = to_2w_series(list(range(len(closes))), closes)
        return cc

    # BTC 2W RSI / Stoch with MA on RSI
    btc_2w = to_2w(cl_w_btc) if cl_w_btc else []
    btc_rsi2w = rsi_from_closes(btc_2w, period=14) if btc_2w else []
    btc_rsi2w_ma = sma(btc_rsi2w, period=9) if btc_rsi2w else []
    btc_k2w, btc_d2w = stoch_rsi_kd(
        btc_2w, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3) if btc_2w else ([], [])

    # ETH/BTC 2W RSI (this is a relative momentum proxy)
    ethbtc_2w = to_2w(cl_w_ethbtc) if cl_w_ethbtc else []
    ethbtc_rsi2w = rsi_from_closes(ethbtc_2w, period=14) if ethbtc_2w else []
    ethbtc_rsi2w_ma = sma(ethbtc_rsi2w, period=9) if ethbtc_rsi2w else []

    # ALT basket 2W RSI/Stoch + simple fib proximity aggregation
    alt_rsi_vals: List[float] = []
    alt_rsi_ma_vals: List[float] = []
    alt_k_vals: List[float] = []
    alt_d_vals: List[float] = []
    alt_fib_prox_vals: List[float] = []

    for inst, cl in alt_closes_map.items():
        two_w = to_2w(cl)
        if len(two_w) >= 20:
            r = rsi_from_closes(two_w, 14)
            rma = sma(r, 9)
            k, d = stoch_rsi_kd(two_w, 14, 14, 3, 3)

            alt_rsi_vals.append(float(r[-1]))
            alt_rsi_ma_vals.append(float(rma[-1]))
            alt_k_vals.append(float(k[-1]))
            alt_d_vals.append(float(d[-1]))

        # Fib proximity on 1W for each ALT, then average
        if len(cl) >= 60:
            s = pd.Series(cl, dtype=float)
            lo = float(s[-60:].min())
            hi = float(s[-60:].max())
            last = float(s.iloc[-1])
            rng = max(1e-9, hi - lo)
            fib1272 = hi + 1.272 * rng
            prox = abs(last - fib1272) / fib1272 * 100.0
            alt_fib_prox_vals.append(prox)

    alt_rsi2w = float(np.mean(alt_rsi_vals)) if alt_rsi_vals else float("nan")
    alt_rsi2w_ma = float(np.mean(alt_rsi_ma_vals)
                         ) if alt_rsi_ma_vals else float("nan")
    alt_k2w = float(np.mean(alt_k_vals)) if alt_k_vals else float("nan")
    alt_d2w = float(np.mean(alt_d_vals)) if alt_d_vals else float("nan")
    alt_fib_prox = float(np.median(alt_fib_prox_vals)
                         ) if alt_fib_prox_vals else float("nan")

    # BTC weekly Fib proximity (reference window 100w)
    btc_fib_prox = float("nan")
    if len(cl_w_btc) >= 100:
        s = pd.Series(cl_w_btc, dtype=float)
        lo = float(s[-100:].min())
        hi = float(s[-100:].max())
        last = float(s.iloc[-1])
        rng = max(1e-9, hi - lo)
        fib1272 = hi + 1.272 * rng
        btc_fib_prox = abs(last - fib1272) / fib1272 * 100.0

    # Build metric dicts
    m: Dict[str, Any] = dict(
        btc_dom=btc_dom,
        eth_btc=eth_btc,
        altcap_btc_ratio=alt_btc_ratio,
        funding_max=funding_max,
        funding_median=funding_median,
        oi_btc=oi_btc_usd,
        oi_eth=oi_eth_usd,
        trends=trends,
        pi_prox=pi_prox,
        btc_rsi2w=(btc_rsi2w[-1] if btc_rsi2w else float("nan")),
        btc_rsi2w_ma=(btc_rsi2w_ma[-1] if btc_rsi2w_ma else float("nan")),
        btc_k2w=(btc_k2w[-1] if btc_k2w else float("nan")),
        btc_d2w=(btc_d2w[-1] if btc_d2w else float("nan")),
        ethbtc_rsi2w=(ethbtc_rsi2w[-1] if ethbtc_rsi2w else float("nan")),
        ethbtc_rsi2w_ma=(ethbtc_rsi2w_ma[-1]
                         if ethbtc_rsi2w_ma else float("nan")),
        btc_fib_prox=btc_fib_prox,
        alt_rsi2w=alt_rsi2w,
        alt_rsi2w_ma=alt_rsi2w_ma,
        alt_k2w=alt_k2w,
        alt_d2w=alt_d2w,
        alt_fib_prox=alt_fib_prox,
    )

    f: Dict[str, Any] = dict(
        today=fng_today,
        ma14=fng_ma14,
        ma30=fng_ma30,
        run_days=fng_run_days,
        pct30=fng_pct30,
    )

    aux: Dict[str, Any] = {
        "funding_rows": funding_rows,
    }

    return m, f, aux

# --------------------------------------------------------------------------------------
# Composite scoring (Alt-Top Certainty)
# --------------------------------------------------------------------------------------


def score_linear(x: float, warn: float, flag: float, direction: str = "above") -> float:
    if math.isnan(x):
        return 0.0
    if direction == "above":
        if x <= warn:
            return 0.0
        if x >= flag:
            return 100.0
        return 100.0 * (x - warn) / (flag - warn)
    else:
        # risk increases as x goes DOWN toward/under warn/flag
        if x >= warn:
            return 0.0
        if x <= flag:
            return 100.0
        return 100.0 * (warn - x) / (warn - flag)


def build_composite(m: Dict[str, Any], f: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    # Individual subscores
    subs: Dict[str, float] = {}

    # Structure
    subs["Alt vs BTC cap"] = score_linear(m.get("altcap_btc_ratio", float(
        "nan")), t["alt_btc_warn"], t["alt_btc_flag"], "above")
    subs["ETH/BTC level"] = score_linear(
        m.get("eth_btc", float("nan")), t["ethbtc_warn"], t["ethbtc_flag"], "above")

    # Derivs
    subs["Funding"] = score_linear(m.get("funding_max", float(
        "nan")), t["funding_warn"], t["funding_flag"], "above")
    subs["BTC OI"] = score_linear(m.get("oi_btc", float(
        "nan")), t["oi_btc_warn"], t["oi_btc_flag"], "above")
    subs["ETH OI"] = score_linear(m.get("oi_eth", float(
        "nan")), t["oi_eth_warn"], t["oi_eth_flag"], "above")

    # Sentiment
    subs["Trends"] = score_linear(m.get("trends", float(
        "nan")), t["trends_warn"], t["trends_flag"], "above")
    subs["F&G today"] = score_linear(
        float(f.get("today") or float("nan")), t["fng_warn"], t["fng_flag"], "above")
    subs["F&G 14d"] = score_linear(float(f.get("ma14") or float(
        "nan")), t["fng14_warn"], t["fng14_flag"], "above")
    subs["F&G 30d"] = score_linear(float(f.get("ma30") or float(
        "nan")), t["fng30_warn"], t["fng30_flag"], "above")
    # Persistence: days & pct both matter; take max
    day_score = score_linear(float(f.get("run_days", 0)), float(
        t["fng_days_warn"]), float(t["fng_days_flag"]), "above")
    pct_score = score_linear(float(f.get("pct30") or float("nan")), float(
        t["fng_pct30_warn"]), float(t["fng_pct30_flag"]), "above")
    subs["F&G persist"] = max(day_score, pct_score)

    # Pi Cycle proximity (closer distance => higher risk)
    pi = m.get("pi_prox")
    if pi is None or math.isnan(pi):
        subs["Pi Cycle prox"] = 0.0
    else:
        # 0% distance => 100 risk, 3% distance => 0 risk
        subs["Pi Cycle prox"] = clamp(
            100.0 * (1.0 - (pi / t["pi_close_pct"])), 0.0, 100.0)

    # Momentum (2W)
    subs["BTC RSI (2W)"] = score_linear(m.get("btc_rsi2w", float(
        "nan")), t["rsi_btc2w_warn"], t["rsi_btc2w_flag"], "above")
    subs["ETH/BTC RSI (2W)"] = score_linear(m.get("ethbtc_rsi2w", float("nan")),
                                            t["rsi_ethbtc2w_warn"], t["rsi_ethbtc2w_flag"], "above")
    # If RSI < its MA, add a small penalty (10 points)
    if not math.isnan(m.get("btc_rsi2w", float("nan"))) and not math.isnan(m.get("btc_rsi2w_ma", float("nan"))):
        if m["btc_rsi2w"] < m["btc_rsi2w_ma"]:
            subs["BTC RSI (2W)"] = clamp(
                subs["BTC RSI (2W)"] + 10.0, 0.0, 100.0)
    if not math.isnan(m.get("ethbtc_rsi2w", float("nan"))) and not math.isnan(m.get("ethbtc_rsi2w_ma", float("nan"))):
        if m["ethbtc_rsi2w"] < m["ethbtc_rsi2w_ma"]:
            subs["ETH/BTC RSI (2W)"] = clamp(subs["ETH/BTC RSI (2W)"] +
                                             10.0, 0.0, 100.0)

    # Stoch overbought + bearish cross risk
    stoch_ob = 0.80
    st_btc = 0.0
    if not math.isnan(m.get("btc_k2w", float("nan"))) and not math.isnan(m.get("btc_d2w", float("nan"))):
        k = m["btc_k2w"]
        d = m["btc_d2w"]
        if k >= stoch_ob:
            st_btc = 60.0
            if k < d:
                st_btc = 100.0
    subs["BTC Stoch (2W)"] = st_btc

    st_alt = 0.0
    if not math.isnan(m.get("alt_k2w", float("nan"))) and not math.isnan(m.get("alt_d2w", float("nan"))):
        k = m["alt_k2w"]
        d = m["alt_d2w"]
        if k >= stoch_ob:
            st_alt = 60.0
            if k < d:
                st_alt = 100.0
    subs["ALT Stoch (2W)"] = st_alt

    # Fib proximity (closer => higher risk)
    for name, prox in [("BTC Fib 1.272", m.get("btc_fib_prox")), ("ALT Fib 1.272", m.get("alt_fib_prox"))]:
        if prox is None or math.isnan(prox):
            subs[name] = 0.0
        else:
            warn = float(t["fib_warn_pct"])
            flag = float(t["fib_flag_pct"])
            # risk rises as prox falls (below thresholds)
            if prox >= warn:
                subs[name] = 0.0
            elif prox <= flag:
                subs[name] = 100.0
            else:
                subs[name] = 100.0 * (warn - prox) / (warn - flag)

    # ALT RSI (2W)
    subs["ALT RSI (2W)"] = score_linear(m.get("alt_rsi2w", float(
        "nan")), t["rsi_alt2w_warn"], t["rsi_alt2w_flag"], "above")
    if not math.isnan(m.get("alt_rsi2w", float("nan"))) and not math.isnan(m.get("alt_rsi2w_ma", float("nan"))):
        if m["alt_rsi2w"] < m["alt_rsi2w_ma"]:
            subs["ALT RSI (2W)"] = clamp(
                subs["ALT RSI (2W)"] + 10.0, 0.0, 100.0)

    # Equal weights across blocks (already roughly balanced); compute mean
    total = int(round(float(np.mean(list(subs.values()))) if subs else 0.0))

    # Top 5 contributors
    sorted_items = sorted(subs.items(), key=lambda kv: kv[1], reverse=True)
    top5 = [(k, int(round(v))) for k, v in sorted_items[:5]]

    return total, top5

# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def format_snapshot(ts_iso: str, profile: str, m: Dict[str, Any], f: Dict[str, Any], aux: Dict[str, Any]) -> str:
    t = profile_thresholds(profile)

    # Colors
    c_btc_dom = flag_color(m.get("btc_dom", float("nan")),
                           t["btc_dom_warn"], t["btc_dom_flag"], "above")
    c_ethbtc = flag_color(m.get("eth_btc", float("nan")),
                          t["ethbtc_warn"], t["ethbtc_flag"], "above")
    c_altbtc = flag_color(m.get("altcap_btc_ratio", float(
        "nan")), t["alt_btc_warn"], t["alt_btc_flag"], "above")

    c_fmax = flag_color(m.get("funding_max", float("nan")),
                        t["funding_warn"], t["funding_flag"], "above")
    c_fmed = flag_color(m.get("funding_median", float("nan")),
                        t["funding_warn"], t["funding_flag"], "above")

    c_oibtc = flag_color(m.get("oi_btc", float("nan")),
                         t["oi_btc_warn"], t["oi_btc_flag"], "above")
    c_oiaeth = flag_color(m.get("oi_eth", float("nan")),
                          t["oi_eth_warn"], t["oi_eth_flag"], "above")

    c_trends = flag_color(m.get("trends", float("nan")),
                          t["trends_warn"], t["trends_flag"], "above")

    c_fng = flag_color(float(f.get("today") or float("nan")),
                       t["fng_warn"], t["fng_flag"], "above")
    c_fng14 = flag_color(float(f.get("ma14") or float("nan")),
                         t["fng14_warn"], t["fng14_flag"], "above")
    c_fng30 = flag_color(float(f.get("ma30") or float("nan")),
                         t["fng30_warn"], t["fng30_flag"], "above")
    # Persistence color: if either days or pct triggers, use worse
    days_color = flag_color(float(f.get("run_days", 0)), float(
        t["fng_days_warn"]), float(t["fng_days_flag"]), "above")
    pct_color = flag_color(float(f.get("pct30") or float("nan")), float(
        t["fng_pct30_warn"]), float(t["fng_pct30_flag"]), "above")
    c_persist = "🔴" if "🔴" in (days_color, pct_color) else (
        "🟡" if "🟡" in (days_color, pct_color) else "🟢")

    # Momentum colors (2W)
    c_rsi_btc = flag_color(m.get("btc_rsi2w", float("nan")),
                           t["rsi_btc2w_warn"], t["rsi_btc2w_flag"], "above")
    c_rsi_ethbtc = flag_color(m.get("ethbtc_rsi2w", float(
        "nan")), t["rsi_ethbtc2w_warn"], t["rsi_ethbtc2w_flag"], "above")
    c_rsi_alt = flag_color(m.get("alt_rsi2w", float("nan")),
                           t["rsi_alt2w_warn"], t["rsi_alt2w_flag"], "above")

    # Stoch indicators shown without strict color other than overbought/cross description
    def stoch_text(k: float, d: float) -> str:
        if math.isnan(k) or math.isnan(d):
            return "n/a"
        return f"{k:.2f}/{d:.2f}"

    # Fib colors (closer to extensions => higher risk so reverse direction)
    def fib_color(px: Optional[float]) -> str:
        if px is None or math.isnan(px):
            return "🟡"
        if px <= t["fib_flag_pct"]:
            return "🔴"
        if px <= t["fib_warn_pct"]:
            return "🟡"
        return "🟢"

    # Pi Cycle proximity
    pi_txt = "n/a"
    if m.get("pi_prox") is not None and not math.isnan(m.get("pi_prox")):
        pi_txt = f"{m['pi_prox']:.2f}% of trigger (100% = cross)"
    c_header = []

    lines: List[str] = []
    lines.append(f"📊 Crypto Market Snapshot — {ts_iso}")
    lines.append(f"Profile: {profile}")
    lines.append("")
    lines.append("Market Structure")
    lines.append(f"• Bitcoin market share of total crypto: {c_btc_dom} {m.get('btc_dom'):.2f}%  (warn ≥ {t['btc_dom_warn']:.2f}%, flag ≥ {t['btc_dom_flag']:.2f}%)" if not math.isnan(
        m.get('btc_dom', float('nan'))) else "• Bitcoin market share of total crypto: 🟡 n/a")
    lines.append(f"• Ether price relative to Bitcoin (ETH/BTC): {c_ethbtc} {m.get('eth_btc'):.5f}  (warn ≥ {t['ethbtc_warn']:.5f}, flag ≥ {t['ethbtc_flag']:.5f})" if not math.isnan(
        m.get('eth_btc', float('nan'))) else "• Ether price relative to Bitcoin (ETH/BTC): 🟡 n/a")
    if not math.isnan(m.get("altcap_btc_ratio", float("nan"))):
        lines.append(
            f"• Altcoin market cap / Bitcoin market cap: {c_altbtc} {m['altcap_btc_ratio']:.2f}  (warn ≥ {t['alt_btc_warn']:.2f}, flag ≥ {t['alt_btc_flag']:.2f})")
    else:
        lines.append("• Altcoin market cap / Bitcoin market cap: 🟡 n/a")

    lines.append("")
    lines.append("Derivatives")
    # Funding basket summary
    fmax = m.get("funding_max", float("nan"))
    fmed = m.get("funding_median", float("nan"))
    basket_names = [s for (s, _) in aux.get("funding_rows", [])]
    lines.append(
        f"• Funding (basket: {', '.join(basket_names)}) — max: {c_fmax} {pretty_pct(fmax)} | median: {c_fmed} {pretty_pct(fmed)}  (warn ≥ {t['funding_warn']:.3f}%, flag ≥ {t['funding_flag']:.3f}%)")
    # Show top-3 extremes
    tops = sorted(aux.get("funding_rows", []), key=lambda kv: (
        kv[1] if kv[1] is not None else -1e9), reverse=True)[:3]
    tops_s = ", ".join(
        [f"{n} {pretty_pct(v)}" for (n, v) in tops if v is not None])
    if tops_s:
        lines.append(f"  Top-3 funding extremes: {tops_s}")
    # OI
    lines.append(
        f"• Bitcoin open interest (USD): {c_oibtc} {pretty_usd(m.get('oi_btc'))}  (warn ≥ {pretty_usd(t['oi_btc_warn'])}, flag ≥ {pretty_usd(t['oi_btc_flag'])})")
    lines.append(
        f"• Ether open interest (USD): {c_oiaeth} {pretty_usd(m.get('oi_eth'))}  (warn ≥ {pretty_usd(t['oi_eth_warn'])}, flag ≥ {pretty_usd(t['oi_eth_flag'])})")

    lines.append("")
    lines.append("Sentiment")
    lines.append(f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {c_trends} {m.get('trends'):.1f}  (warn ≥ {t['trends_warn']:.1f}, flag ≥ {t['trends_flag']:.1f})" if not math.isnan(
        m.get('trends', float('nan'))) else "• Google Trends avg (7d; crypto/bitcoin/ethereum): 🟡 n/a")
    lines.append(
        f"• Fear & Greed Index (overall crypto): {c_fng} {f.get('today') if f.get('today') is not None else 'n/a'}  (warn ≥ {int(t['fng_warn'])}, flag ≥ {int(t['fng_flag'])})")
    lines.append(f"• Fear & Greed 14-day average: {c_fng14} {f.get('ma14'):.1f}  (warn ≥ {int(t['fng14_warn'])}, flag ≥ {int(t['fng14_flag'])})" if f.get(
        "ma14") is not None else "• Fear & Greed 14-day average: 🟡 n/a")
    lines.append(f"• Fear & Greed 30-day average: {c_fng30} {f.get('ma30'):.1f}  (warn ≥ {int(t['fng30_warn'])}, flag ≥ {int(t['fng30_flag'])})" if f.get(
        "ma30") is not None else "• Fear & Greed 30-day average: 🟡 n/a")
    if f.get("pct30") is not None:
        lines.append(
            f"• Greed persistence: {c_persist} {f.get('run_days', 0)} days in a row | {f['pct30']:.0f}% of last 30 days ≥ 70  (warn: days ≥ {t['fng_days_warn']} or pct ≥ {int(t['fng_pct30_warn'])}%; flag: days ≥ {t['fng_days_flag']} or pct ≥ {int(t['fng_pct30_flag'])}%)")
    else:
        lines.append("• Greed persistence: 🟡 n/a")

    lines.append("")
    lines.append("Cycle & On-Chain")
    lines.append(
        f"• Pi Cycle Top proximity: {'🟡 ' + pi_txt if pi_txt=='n/a' else ('🟢 ' + pi_txt if m.get('pi_prox', 99) > 3 else '🔴 ' + pi_txt)}")

    lines.append("")
    lines.append("Momentum (2W) & Extensions (1W)")
    # BTC RSI (2W)
    if not math.isnan(m.get("btc_rsi2w", float("nan"))) and not math.isnan(m.get("btc_rsi2w_ma", float("nan"))):
        lines.append(
            f"• BTC RSI (2W): {c_rsi_btc} {m['btc_rsi2w']:.1f} (MA {m['btc_rsi2w_ma']:.1f}) (warn ≥ {t['rsi_btc2w_warn']:.1f}, flag ≥ {t['rsi_btc2w_flag']:.1f})")
    else:
        lines.append("• BTC RSI (2W): 🟡 n/a")
    # ETH/BTC RSI (2W)
    if not math.isnan(m.get("ethbtc_rsi2w", float("nan"))) and not math.isnan(m.get("ethbtc_rsi2w_ma", float("nan"))):
        lines.append(
            f"• ETH/BTC RSI (2W): {c_rsi_ethbtc} {m['ethbtc_rsi2w']:.1f} (MA {m['ethbtc_rsi2w_ma']:.1f}) (warn ≥ {t['rsi_ethbtc2w_warn']:.1f}, flag ≥ {t['rsi_ethbtc2w_flag']:.1f})")
    else:
        lines.append("• ETH/BTC RSI (2W): 🟡 n/a")
    # ALT basket RSI (2W)
    if not math.isnan(m.get("alt_rsi2w", float("nan"))) and not math.isnan(m.get("alt_rsi2w_ma", float("nan"))):
        lines.append(
            f"• ALT basket (equal-weight) RSI (2W): {c_rsi_alt} {m['alt_rsi2w']:.1f} (MA {m['alt_rsi2w_ma']:.1f}) (warn ≥ {t['rsi_alt2w_warn']:.1f}, flag ≥ {t['rsi_alt2w_flag']:.1f})")
    else:
        lines.append("• ALT basket (equal-weight) RSI (2W): 🟡 n/a")

    # Stoch RSI (2W)
    lines.append(
        f"• BTC Stoch RSI (2W) K/D: {stoch_text(m.get('btc_k2w', float('nan')), m.get('btc_d2w', float('nan')))} (overbought ≥ 0.80; red = bearish cross from OB)")
    lines.append(
        f"• ALT basket Stoch RSI (2W) K/D: {stoch_text(m.get('alt_k2w', float('nan')), m.get('alt_d2w', float('nan')))} (overbought ≥ 0.80; red = bearish cross from OB)")

    # Fibs
    if not math.isnan(m.get("btc_fib_prox", float("nan"))):
        lines.append(
            f"• BTC Fibonacci extension proximity: {fib_color(m.get('btc_fib_prox'))} 1.272 @ {m['btc_fib_prox']:.2f}% away (warn ≤ {t['fib_warn_pct']:.1f}%, flag ≤ {t['fib_flag_pct']:.1f}%)")
    else:
        lines.append("• BTC Fibonacci extension proximity: 🟡 n/a")
    if not math.isnan(m.get("alt_fib_prox", float("nan"))):
        lines.append(
            f"• ALT basket Fibonacci proximity: {fib_color(m.get('alt_fib_prox'))} 1.272 @ {m['alt_fib_prox']:.2f}% away (warn ≤ {t['fib_warn_pct']:.1f}%, flag ≤ {t['fib_flag_pct']:.1f}%)")
    else:
        lines.append("• ALT basket Fibonacci proximity: 🟡 n/a")

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Telegram bot commands
# --------------------------------------------------------------------------------------
DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "moderate")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Hey! I’ll keep an eye on market cycle risk and ping you automatically.\n\n"
        "Commands:\n"
        "• /status — current snapshot\n"
        "• /assess [conservative|moderate|aggressive] — snapshot using that risk profile (default: moderate)\n"
        "\n"
        "Settings via env:\n"
        "• PUSH_HOUR_UTC (default 15) — daily summary time (UTC)\n"
        "• ALERT_EVERY_MIN (default 15) — alert cadence\n"
    )
    await update.message.reply_text(msg)


async def build_and_send_snapshot(app: Application, chat_id: int, profile: str) -> None:
    async with DataClient() as dc:  # type: ignore
        try:
            m, f, aux = await gather_metrics(dc, profile)
            t = profile_thresholds(profile)
            # Composite
            total, top5 = build_composite(m, f, t)
            header = [
                f"📊 Crypto Market Snapshot — {pd.Timestamp.utcnow().isoformat()}Z",
                f"Profile: {profile}",
                ""
            ]
            body = format_snapshot(
                pd.Timestamp.utcnow().isoformat(), profile, m, f, aux)

            # Composite block
            comp_lines = []
            comp_lines.append("")
            comp_lines.append("Alt-Top Certainty (Composite)")
            comp_lines.append(
                f"• Certainty: {traffic_from_score(total)} {total}/100 (yellow ≥ 40, red ≥ 70)")
            if top5:
                comp_lines.append("• Top drivers:")
                for name, sc in top5:
                    comp_lines.append(
                        f"  • {name}: {traffic_from_score(sc)} {sc}/100")

            text = "\n".join((body, "\n".join(comp_lines)))

            for chunk in split_long_message(text):
                await app.bot.send_message(chat_id=chat_id, text=chunk)
        except Exception as e:
            log.exception("snapshot failed")
            await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Could not fetch metrics right now: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    profile = DEFAULT_PROFILE
    if context.args:
        arg = context.args[0].strip().lower()
        if arg in {"conservative", "moderate", "aggressive"}:
            profile = arg
    await build_and_send_snapshot(context.application, update.effective_chat.id, profile)


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    profile = DEFAULT_PROFILE
    if context.args:
        arg = context.args[0].strip().lower()
        if arg in {"conservative", "moderate", "aggressive"}:
            profile = arg
    await build_and_send_snapshot(context.application, update.effective_chat.id, profile)

# --------------------------------------------------------------------------------------
# Push jobs (scheduler)
# --------------------------------------------------------------------------------------
# set this to auto-push somewhere
DAILY_CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", "0"))


async def push_summary(app: Application) -> None:
    if DAILY_CHAT_ID == 0:
        return
    await build_and_send_snapshot(app, DAILY_CHAT_ID, DEFAULT_PROFILE)


async def push_alerts(app: Application) -> None:
    # Light check; if any hard flags (e.g., F&G persist, Pi very close), send a compact ping
    if DAILY_CHAT_ID == 0:
        return
    async with DataClient() as dc:  # type: ignore
        try:
            m, f, aux = await gather_metrics(dc, DEFAULT_PROFILE)
            t = profile_thresholds(DEFAULT_PROFILE)
            total, top5 = build_composite(m, f, t)
            flags: List[str] = []

            if f.get("today") is not None and f["today"] >= t["fng_flag"]:
                flags.append("Fear & Greed (today) in Greed")
            if f.get("ma30") is not None and f["ma30"] >= t["fng30_flag"]:
                flags.append("F&G 30-day avg in Greed")
            if f.get("pct30") is not None and f["pct30"] >= t["fng_pct30_flag"]:
                flags.append("Greed in ≥60% of last 30d")
            if m.get("pi_prox") is not None and not math.isnan(m.get("pi_prox")) and m["pi_prox"] <= t["pi_close_pct"]:
                flags.append("Pi Cycle proximity close")

            if total >= 70 or flags:
                lines = ["⚠️ Cycle-risk alert"]
                lines.append(
                    f"• Composite certainty: {traffic_from_score(total)} {total}/100")
                if flags:
                    lines.append("• Triggers: " + "; ".join(flags))
                if top5:
                    lines.append("• Top drivers:")
                    for name, sc in top5:
                        lines.append(
                            f"  • {name}: {traffic_from_score(sc)} {sc}/100")
                await contextless_send(app, DAILY_CHAT_ID, "\n".join(lines))
        except Exception as e:
            log.warning("push_alerts failed: %s", e)


async def contextless_send(app: Application, chat_id: int, text: str) -> None:
    for chunk in split_long_message(text):
        await app.bot.send_message(chat_id=chat_id, text=chunk)

# --------------------------------------------------------------------------------------
# Health server
# --------------------------------------------------------------------------------------


async def _health(_request):
    return web.json_response({"status": "ok"})


async def start_health_server(host="0.0.0.0", port=8080):
    app = web.Application()
    app.router.add_get("/health", _health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    log.info("Health server listening on :%s", port)
    return runner

# --------------------------------------------------------------------------------------
# Main (single event loop; no run_polling)
# --------------------------------------------------------------------------------------


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")

    app = Application.builder().token(token).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("assess", cmd_assess))

    # Health server
    port = int(os.getenv("PORT", "8080"))
    health_runner = await start_health_server("0.0.0.0", port)

    # Start Telegram (no run_polling)
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("Bot running. Press Ctrl+C to exit.")

    # Scheduler
    loop = asyncio.get_running_loop()
    scheduler = AsyncIOScheduler(event_loop=loop, timezone="UTC")
    push_hour = int(os.getenv("PUSH_HOUR_UTC", "15"))
    alert_every = int(os.getenv("ALERT_EVERY_MIN", "15"))
    scheduler.add_job(push_summary, trigger="cron", hour=push_hour, minute=0, args=[
                      app], id="push_summary", replace_existing=True)
    scheduler.add_job(push_alerts, trigger="cron", minute=f"*/{alert_every}", args=[
                      app], id="push_alerts", replace_existing=True)
    scheduler.start()
    log.info("Scheduler started")

    # Graceful shutdown
    stop_event = asyncio.Event()

    def _stop():
        try:
            stop_event.set()
        except Exception:
            pass
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    await stop_event.wait()

    # Teardown
    scheduler.shutdown(wait=False)
    await app.updater.stop()
    await app.stop()
    await app.shutdown()
    await health_runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
