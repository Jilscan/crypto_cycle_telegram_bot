# main.py
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from aiohttp import web
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import numpy as np
import pandas as pd

from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# ---------- Logging ----------
LOG = logging.getLogger("crypto-cycle-bot")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)

# ---------- Globals ----------
LATEST_CHAT_ID: Optional[int] = None  # set on first user interaction

# ---------- Helpers: formatting ----------


def fmt_pct(x: Optional[float], decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{decimals}f}%"


def fmt_usd(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    v = float(x)
    if abs(v) >= 1e12:
        return f"${v/1e12:.3f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.3f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.3f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.3f}K"
    return f"${v:,.2f}"


def fmt_ratio(x: Optional[float], decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "n/a"
    return f"{x:.{decimals}f}"


def color_flag(value: Optional[float], warn: float, flag: float, higher_is_worse: bool = True) -> str:
    """Return color emoji for the value based on thresholds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "🟡"  # neutral for n/a
    v = float(value)
    if higher_is_worse:
        if v >= flag:
            return "🔴"
        if v >= warn:
            return "🟡"
        return "🟢"
    else:
        # lower values are worse (e.g., Pi proximity)
        if v <= flag:
            return "🔴"
        if v <= warn:
            return "🟡"
        return "🟢"

# ---------- Indicators ----------


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    s = series.astype(float).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / (loss.replace(0, np.nan))
    r = 100 - (100 / (1 + rs))
    return r.reindex(series.index)


def stoch_rsi(series: pd.Series, length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    r = rsi(series, length)
    r_min = r.rolling(length).min()
    r_max = r.rolling(length).max()
    stoch = (r - r_min) / (r_max - r_min)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k, d


def resample_to_2w(closes: List[Tuple[int, float]]) -> pd.Series:
    """closes: list of (millis, close) ascending"""
    if not closes:
        return pd.Series(dtype=float)
    df = pd.DataFrame(closes, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    # resample to 2-week periods (ending on Sunday by default)
    s2w = df["close"].resample("2W").last().dropna()
    return s2w


def fib_1272_proximity(weekly_closes: List[Tuple[int, float]]) -> Optional[float]:
    """Distance (pct) from current to 1.272 extension based on last ~52w swing."""
    if not weekly_closes:
        return None
    df = pd.DataFrame(weekly_closes, columns=["ts", "close"])
    df = df.sort_values("ts")
    s = pd.Series(df["close"].astype(float).values)
    if len(s) < 60:
        return None
    # simple lookback
    window = s[-60:]
    lo = float(window.min())
    hi = float(window.max())
    cur = float(s.iloc[-1])
    if hi <= lo:
        return None
    # assume uptrend if last close >= mid
    if cur >= (lo + hi) / 2:
        ext = hi + (hi - lo) * 0.272
    else:
        # downtrend case: extension below
        ext = lo - (hi - lo) * 0.272
    denom = max(1e-12, abs(ext))
    pct = abs(cur - ext) / denom * 100.0
    return pct

# ---------- Thresholds ----------


def profile_thresholds(profile: str) -> Dict[str, Any]:
    p = (profile or "").lower().strip()
    if p not in {"conservative", "moderate", "aggressive"}:
        p = "moderate"
    scale = {"conservative": 0.9, "moderate": 1.0, "aggressive": 1.1}[p]
    t = {
        "btc_dom_warn": 48.0 * (1/scale),
        "btc_dom_flag": 60.0 * (1/scale),
        "ethbtc_warn": 0.072 * scale,
        "ethbtc_flag": 0.090 * scale,
        "alt_btc_warn": 1.44 * scale,
        "alt_btc_flag": 1.80 * scale,

        # funding thresholds are percentage values (e.g., 0.08%):
        "funding_warn": 0.08 * scale,
        "funding_flag": 0.10 * scale,

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

        "stoch_ob": 0.80,
        "fib_warn_pct": 3.0 / scale,
        "fib_flag_pct": 1.5 / scale,

        # Pi: lower is worse (closeness to cross, percent)
        "pi_close_warn": 3.0,
        "pi_close_flag": 1.0,
    }
    return t

# ---------- Data client ----------


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(15.0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    # ---- OKX helpers ----
    async def okx_get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"https://www.okx.com/api/v5{path}"
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def okx_funding(self, inst_id: str) -> Optional[float]:
        """Return funding rate in PERCENT (e.g., 0.010%)."""
        j = await self.okx_get("/public/funding-rate", {"instId": inst_id})
        data = j.get("data", [])
        if not data:
            return None
        # OKX returns as fraction (e.g., 0.0001 = 0.01%)
        rate = float(data[0].get("fundingRate", "0"))
        return rate * 100.0

    async def okx_oi_value(self, inst_id: str) -> Optional[float]:
        j = await self.okx_get("/public/open-interest", {"instType": "SWAP", "instId": inst_id})
        data = j.get("data", [])
        if not data:
            return None
        v = data[0].get("oiValue")
        return float(v) if v is not None else None

    async def okx_ticker_last(self, inst_id: str) -> Optional[float]:
        j = await self.okx_get("/market/ticker", {"instId": inst_id})
        data = j.get("data", [])
        if not data:
            return None
        last = data[0].get("last")
        return float(last) if last is not None else None

    async def okx_candles(self, inst_id: str, bar: str = "1W", limit: int = 400) -> List[Tuple[int, float]]:
        j = await self.okx_get("/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})
        data = j.get("data", [])
        out: List[Tuple[int, float]] = []
        # OKX returns newest first; reverse to ascending
        for row in reversed(data):
            ts_ms = int(row[0])
            close = float(row[4])
            out.append((ts_ms, close))
        return out

    # ---- Google Trends (pytrends) ----
    async def google_trends_avg(self, keywords: List[str]) -> Optional[float]:
        def _fetch() -> Optional[float]:
            try:
                from pytrends.request import TrendReq
                pt = TrendReq(hl="en-US", tz=0)
                pt.build_payload(keywords, timeframe="now 7-d")
                df = pt.interest_over_time()
                if df.empty:
                    return None
                # drop 'isPartial' if present
                df = df.drop(
                    columns=[c for c in df.columns if c.lower() == "ispartial"], errors="ignore")
                vals = df.values.astype(float)
                if vals.size == 0:
                    return None
                return float(np.nanmean(vals))
            except Exception as e:
                LOG.warning("google_trends_avg failed: %s", e)
                return None
        return await asyncio.to_thread(_fetch)

    # ---- Fear & Greed (alternative.me) ----
    async def fear_greed(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            # today
            r1 = await self.client.get("https://api.alternative.me/fng/", params={"limit": 1})
            r1.raise_for_status()
            j1 = r1.json()
            d1 = j1.get("data", [])
            if d1:
                out["today"] = int(d1[0].get("value", "0"))
            # last 60 for averages & persistence
            r2 = await self.client.get("https://api.alternative.me/fng/", params={"limit": 60})
            r2.raise_for_status()
            j2 = r2.json()
            d2 = j2.get("data", [])
            vals = [int(x.get("value", "0")) for x in d2][::-1]  # ascending
            s = pd.Series(vals, dtype=float)
            out["ma14"] = float(s.rolling(14).mean().iloc[-1]
                                ) if len(s) >= 14 else None
            out["ma30"] = float(s.rolling(30).mean().iloc[-1]
                                ) if len(s) >= 30 else None
            # persistence ≥70
            ge70 = s >= 70
            days_row = 0
            for hit in reversed(list(ge70)):
                if hit:
                    days_row += 1
                else:
                    break
            out["persist_days"] = int(days_row)
            last30 = ge70.iloc[-30:] if len(ge70) >= 30 else ge70
            out["persist_pct30"] = float(
                100.0 * last30.mean()) if len(last30) > 0 else None
        except Exception as e:
            LOG.warning("fear_greed failed: %s", e)
        return out

    # ---- Coingecko global (only this endpoint; the chart API often 401/429) ----
    async def cg_global(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            r = await self.client.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            j = r.json().get("data", {})
            mkt = j.get("market_cap_percentage", {})
            btc_dom = float(mkt.get("btc", 0.0))
            out["btc_dominance"] = btc_dom
            # total market & btc market to compute altcap/btc ratio if possible
            total_mkt = j.get("total_market_cap", {}).get("usd")
            btc_mkt = j.get("market_cap_percentage", {}).get("btc")
            # j doesn't include BTC market cap in USD directly; compute altcap/btc later using other inputs
            out["total_mkt_usd"] = float(
                total_mkt) if total_mkt is not None else None
        except Exception as e:
            LOG.warning("cg_global failed: %s", e)
        return out


# ---------- Metric collection ----------
async def gather_metrics(profile: str) -> Dict[str, Any]:
    t = profile_thresholds(profile)
    m: Dict[str, Any] = {}

    async with DataClient() as dc:
        # Market structure
        g = await dc.cg_global()
        btc_dom = g.get("btc_dominance")
        m["btc_dominance"] = btc_dom

        # ETH/BTC weekly RSI (2W later), but we still need spot ratio right now:
        ethbtc_week = await dc.okx_candles("ETH-BTC", "1W", 400)
        if ethbtc_week:
            m["eth_btc"] = float(ethbtc_week[-1][1])
        else:
            m["eth_btc"] = None

        # Altcap/BTC ratio proxy:
        # Use total crypto market cap from cg_global (if available) and approximate BTC mkt dominance
        # alt/btc ratio ≈ (Total/BTC) - 1  -> with dominance in %
        if g.get("total_mkt_usd") and btc_dom:
            # BTC share percent -> BTC cap = total * btc_dom%
            # altcap = total - btc_cap; ratio = altcap / btc_cap
            total = float(g["total_mkt_usd"])
            btc_cap = total * (float(btc_dom) / 100.0)
            alt_cap = max(0.0, total - btc_cap)
            m["altcap_btc_ratio"] = (
                alt_cap / btc_cap) if btc_cap > 0 else None
        else:
            m["altcap_btc_ratio"] = None

        # Derivatives: funding & OI from OKX (USDT perpetuals)
        basket = ["BTC-USDT-SWAP", "ETH-USDT-SWAP",
                  "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]
        basket_funding: List[Tuple[str, Optional[float]]] = []
        for inst in basket:
            try:
                f = await dc.okx_funding(inst)
            except Exception as e:
                LOG.warning("funding %s failed: %s", inst, e)
                f = None
            basket_funding.append(
                (inst.replace("-SWAP", "").replace("-", ""), f))
        # stats
        fund_vals = [v for _, v in basket_funding if v is not None]
        m["funding_max"] = float(max(fund_vals)) if fund_vals else None
        m["funding_median"] = float(
            np.median(fund_vals)) if fund_vals else None
        # top 3 by absolute
        basket_sorted = sorted([x for x in basket_funding if x[1]
                               is not None], key=lambda kv: abs(kv[1]), reverse=True)
        m["funding_top"] = basket_sorted[:3]

        # OI (USD value) BTC/ETH
        try:
            m["oi_btc_usd"] = await dc.okx_oi_value("BTC-USDT-SWAP")
        except Exception as e:
            LOG.warning("oi btc failed: %s", e)
            m["oi_btc_usd"] = None
        try:
            m["oi_eth_usd"] = await dc.okx_oi_value("ETH-USDT-SWAP")
        except Exception as e:
            LOG.warning("oi eth failed: %s", e)
            m["oi_eth_usd"] = None

        # Sentiment: Google Trends + F&G (+ persistence)
        m["trends_avg"] = await dc.google_trends_avg(["crypto", "bitcoin", "ethereum"])
        fng = await dc.fear_greed()
        m["fng_today"] = fng.get("today")
        m["fng_ma14"] = fng.get("ma14")
        m["fng_ma30"] = fng.get("ma30")
        m["fng_persist_days"] = fng.get("persist_days")
        m["fng_persist_pct30"] = fng.get("persist_pct30")

        # Cycle: Pi proximity from OKX daily
        btc_day = await dc.okx_candles("BTC-USDT", "1D", 500)
        pi_prox = None
        if btc_day and len(btc_day) >= 350:
            closes = [c for _, c in btc_day]
            s = pd.Series(closes, dtype=float)
            sma111 = s.rolling(window=111, min_periods=111).mean()
            sma350 = s.rolling(window=350, min_periods=350).mean()
            both = pd.concat(
                [sma111.rename("s111"), sma350.rename("s350")], axis=1).dropna()
            if not both.empty:
                v111 = float(both.iloc[-1]["s111"])
                v350 = float(both.iloc[-1]["s350"])
                denom = max(1e-12, 2.0 * v350)
                pi_prox = abs(v111 - 2.0 * v350) / denom * 100.0
        m["pi_prox"] = pi_prox

        # Momentum (2W): BTC, ETH/BTC, ALT basket
        # BTC weekly -> 2W
        btc_week = await dc.okx_candles("BTC-USDT", "1W", 400)
        ethbtc_week = ethbtc_week or await dc.okx_candles("ETH-BTC", "1W", 400)

        btc_2w = resample_to_2w(btc_week)
        btc_rsi2w = rsi(btc_2w, 14)
        m["btc_rsi2w"] = float(
            btc_rsi2w.iloc[-1]) if len(btc_rsi2w.dropna()) else None
        m["btc_rsi2w_ma"] = float(btc_rsi2w.rolling(
            9).mean().iloc[-1]) if len(btc_rsi2w.dropna()) >= 9 else None
        k_btc, d_btc = stoch_rsi(btc_2w, 14, 3, 3)
        m["btc_stoch2w_k"] = float(
            k_btc.iloc[-1]) if len(k_btc.dropna()) else None
        m["btc_stoch2w_d"] = float(
            d_btc.iloc[-1]) if len(d_btc.dropna()) else None

        ethbtc_2w = resample_to_2w(ethbtc_week)
        ethbtc_rsi2w = rsi(ethbtc_2w, 14)
        m["ethbtc_rsi2w"] = float(
            ethbtc_rsi2w.iloc[-1]) if len(ethbtc_rsi2w.dropna()) else None
        m["ethbtc_rsi2w_ma"] = float(ethbtc_rsi2w.rolling(
            9).mean().iloc[-1]) if len(ethbtc_rsi2w.dropna()) >= 9 else None

        # ALT basket momentum (2W): equal-weight RSI & StochRSI
        alt_syms = ["SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
                    "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"]
        alt_rsis: List[float] = []
        alt_ks: List[float] = []
        alt_ds: List[float] = []
        for sym in alt_syms:
            try:
                w = await dc.okx_candles(sym, "1W", 400)
                s2 = resample_to_2w(w)
                r = rsi(s2, 14)
                if len(r.dropna()):
                    alt_rsis.append(float(r.iloc[-1]))
                k, d = stoch_rsi(s2, 14, 3, 3)
                if len(k.dropna()):
                    alt_ks.append(float(k.iloc[-1]))
                if len(d.dropna()):
                    alt_ds.append(float(d.iloc[-1]))
            except Exception:
                continue
        m["alt_rsi2w"] = float(np.mean(alt_rsis)) if alt_rsis else None
        m["alt_stoch2w_k"] = float(np.mean(alt_ks)) if alt_ks else None
        m["alt_stoch2w_d"] = float(np.mean(alt_ds)) if alt_ds else None
        # 9-period MA for alt_rsi2w (approximate using BTC 2W index for alignment is overkill; skip MA for basket)

        # Fibonacci proximity (1W) BTC & ALT basket proxy (average of symbols)
        m["btc_fib1272_prox"] = fib_1272_proximity(btc_week)
        alt_fibs: List[float] = []
        for sym in alt_syms:
            try:
                w = await dc.okx_candles(sym, "1W", 400)
                p = fib_1272_proximity(w)
                if p is not None:
                    alt_fibs.append(p)
            except Exception:
                continue
        m["alt_fib1272_prox"] = float(np.mean(alt_fibs)) if alt_fibs else None

    return m

# ---------- Composite scoring ----------


def linear_score(value: Optional[float], warn: float, flag: float, higher_is_worse: bool = True) -> float:
    """Return 0..100 severity score based on linear ramp from warn to flag."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    v = float(value)
    if higher_is_worse:
        if v <= warn:
            return 0.0
        if v >= flag:
            return 100.0
        return (v - warn) / max(1e-12, (flag - warn)) * 100.0
    else:
        # lower worse
        if v >= warn:
            return 0.0
        if v <= flag:
            return 100.0
        return (warn - v) / max(1e-12, (warn - flag)) * 100.0


def composite_certainty(m: Dict[str, Any], t: Dict[str, Any]) -> Tuple[int, List[Tuple[str, float]]]:
    """Return (0..100 certainty, top_contributors list[(label, score)])"""
    subs: List[Tuple[str, float, float]] = []  # (label, raw 0..100, weight)

    # Market structure
    if m.get("altcap_btc_ratio") is not None:
        subs.append(("Altcap/BTC ratio", linear_score(
            m["altcap_btc_ratio"], t["alt_btc_warn"], t["alt_btc_flag"], True), 0.08))
    if m.get("eth_btc") is not None:
        subs.append(
            ("ETH/BTC", linear_score(m["eth_btc"], t["ethbtc_warn"], t["ethbtc_flag"], True), 0.06))

    # Derivatives
    if m.get("funding_max") is not None:
        subs.append(("Funding (max)", linear_score(
            m["funding_max"], t["funding_warn"], t["funding_flag"], True), 0.08))
    if m.get("oi_btc_usd") is not None:
        subs.append(("BTC OI", linear_score(
            m["oi_btc_usd"], t["oi_btc_warn"], t["oi_btc_flag"], True), 0.10))
    if m.get("oi_eth_usd") is not None:
        subs.append(("ETH OI", linear_score(
            m["oi_eth_usd"], t["oi_eth_warn"], t["oi_eth_flag"], True), 0.06))

    # Sentiment
    if m.get("trends_avg") is not None:
        subs.append(("Google Trends", linear_score(
            m["trends_avg"], t["trends_warn"], t["trends_flag"], True), 0.06))
    if m.get("fng_today") is not None:
        subs.append(("F&G (today)", linear_score(
            m["fng_today"], t["fng_warn"], t["fng_flag"], True), 0.07))
    if m.get("fng_ma14") is not None:
        subs.append(("F&G 14d", linear_score(
            m["fng_ma14"], t["fng14_warn"], t["fng14_flag"], True), 0.07))
    if m.get("fng_ma30") is not None:
        subs.append(("F&G 30d", linear_score(
            m["fng_ma30"], t["fng30_warn"], t["fng30_flag"], True), 0.08))
    if m.get("fng_persist_days") is not None and m.get("fng_persist_pct30") is not None:
        # combine persistence: either many consecutive days or high 30d %
        days_score = linear_score(float(m["fng_persist_days"]), float(
            t["fng_days_warn"]), float(t["fng_days_flag"]), True)
        pct_score = linear_score(float(m["fng_persist_pct30"]), float(
            t["fng_pct30_warn"]), float(t["fng_pct30_flag"]), True)
        subs.append(("F&G persistence", max(days_score, pct_score), 0.08))

    # Cycle / On-chain
    if m.get("pi_prox") is not None:
        subs.append(("Pi proximity", linear_score(
            m["pi_prox"], t["pi_close_warn"], t["pi_close_flag"], False), 0.06))

    # Momentum (2W)
    if m.get("btc_rsi2w") is not None:
        subs.append(("BTC RSI (2W)", linear_score(
            m["btc_rsi2w"], t["rsi_btc2w_warn"], t["rsi_btc2w_flag"], True), 0.07))
    if m.get("ethbtc_rsi2w") is not None:
        subs.append(("ETH/BTC RSI (2W)", linear_score(
            m["ethbtc_rsi2w"], t["rsi_ethbtc2w_warn"], t["rsi_ethbtc2w_flag"], True), 0.05))
    if m.get("alt_rsi2w") is not None:
        subs.append(("ALT basket RSI (2W)", linear_score(
            m["alt_rsi2w"], t["rsi_alt2w_warn"], t["rsi_alt2w_flag"], True), 0.05))
    if m.get("btc_stoch2w_k") is not None and m.get("btc_stoch2w_d") is not None:
        # score if overbought and crossing down: approximate via K>OB and K<D
        k = float(m["btc_stoch2w_k"])
        d = float(m["btc_stoch2w_d"])
        st_score = 100.0 if (k >= t["stoch_ob"] and k < d) else (
            50.0 if k >= t["stoch_ob"] else 0.0)
        subs.append(("BTC StochRSI (2W)", st_score, 0.03))
    if m.get("alt_stoch2w_k") is not None and m.get("alt_stoch2w_d") is not None:
        k = float(m["alt_stoch2w_k"])
        d = float(m["alt_stoch2w_d"])
        st_score = 100.0 if (k >= t["stoch_ob"] and k < d) else (
            50.0 if k >= t["stoch_ob"] else 0.0)
        subs.append(("ALT StochRSI (2W)", st_score, 0.03))

    # Extensions proximity (1W)
    if m.get("btc_fib1272_prox") is not None:
        subs.append(("BTC Fib 1.272", linear_score(
            m["btc_fib1272_prox"], t["fib_warn_pct"], t["fib_flag_pct"], False), 0.03))
    if m.get("alt_fib1272_prox") is not None:
        subs.append(("ALT Fib 1.272", linear_score(
            m["alt_fib1272_prox"], t["fib_warn_pct"], t["fib_flag_pct"], False), 0.03))

    # Weighted sum
    total_w = sum(w for _, _, w in subs) or 1.0
    weighted = sum(score * w for _, score, w in subs) / total_w
    certainty = int(round(min(100.0, max(0.0, weighted))))

    # Top 5 contributors by *impact* (score*weight)
    impacts = sorted([(label, score * w) for (label, score, w)
                     in subs], key=lambda x: x[1], reverse=True)
    top5 = [(label, round(min(100.0, max(0.0, impact * (100.0 / 100.0))), 1))
            for (label, impact) in impacts[:5]]
    # Note: we show the weighted contribution normalized to 0..100 scale notionally

    return certainty, top5

# ---------- Snapshot formatting ----------


def build_snapshot_text(m: Dict[str, Any], profile: str) -> str:
    t = profile_thresholds(profile)
    now = datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    lines: List[str] = []
    lines.append(f"📊 Crypto Market Snapshot — {now}")
    lines.append(f"Profile: {profile}")
    lines.append("")

    # Market Structure
    lines.append("Market Structure")
    btc_dom = m.get("btc_dominance")
    lines.append(
        f"• Bitcoin market share of total crypto: "
        f"{color_flag(btc_dom, t['btc_dom_warn'], t['btc_dom_flag'], True)} "
        f"{fmt_ratio(btc_dom, 2)}%  (warn ≥ {t['btc_dom_warn']:.2f}%, flag ≥ {t['btc_dom_flag']:.2f}%)"
    )
    eth_btc = m.get("eth_btc")
    lines.append(
        f"• Ether price relative to Bitcoin (ETH/BTC): "
        f"{color_flag(eth_btc, t['ethbtc_warn'], t['ethbtc_flag'], True)} "
        f"{fmt_ratio(eth_btc, 5)}  (warn ≥ {t['ethbtc_warn']:.5f}, flag ≥ {t['ethbtc_flag']:.5f})"
    )
    alt_ratio = m.get("altcap_btc_ratio")
    lines.append(
        f"• Altcoin market cap / Bitcoin market cap: "
        f"{color_flag(alt_ratio, t['alt_btc_warn'], t['alt_btc_flag'], True)} "
        f"{fmt_ratio(alt_ratio, 2)}  (warn ≥ {t['alt_btc_warn']:.2f}, flag ≥ {t['alt_btc_flag']:.2f})"
    )
    lines.append("")

    # Derivatives
    lines.append("Derivatives")
    fund_max = m.get("funding_max")
    fund_med = m.get("funding_median")
    lines.append(
        f"• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — "
        f"max: {color_flag(fund_max, t['funding_warn'], t['funding_flag'], True)} {fmt_pct(fund_max, 3)} | "
        f"median: {color_flag(fund_med, t['funding_warn'], t['funding_flag'], True)} {fmt_pct(fund_med, 3)}  "
        f"(warn ≥ {t['funding_warn']:.3f}%, flag ≥ {t['funding_flag']:.3f}%)"
    )
    top_ex = m.get("funding_top") or []
    if top_ex:
        top_txt = ", ".join(
            [f"{sym} {fmt_pct(val, 3)}" for sym, val in top_ex])
        lines.append(f"  Top-3 funding extremes: {top_txt}")
    oi_btc = m.get("oi_btc_usd")
    lines.append(
        f"• Bitcoin open interest (USD): "
        f"{color_flag(oi_btc, t['oi_btc_warn'], t['oi_btc_flag'], True)} {fmt_usd(oi_btc)}  "
        f"(warn ≥ {fmt_usd(t['oi_btc_warn'])}, flag ≥ {fmt_usd(t['oi_btc_flag'])})"
    )
    oi_eth = m.get("oi_eth_usd")
    lines.append(
        f"• Ether open interest (USD): "
        f"{color_flag(oi_eth, t['oi_eth_warn'], t['oi_eth_flag'], True)} {fmt_usd(oi_eth)}  "
        f"(warn ≥ {fmt_usd(t['oi_eth_warn'])}, flag ≥ {fmt_usd(t['oi_eth_flag'])})"
    )
    lines.append("")

    # Sentiment
    lines.append("Sentiment")
    trends = m.get("trends_avg")
    lines.append(
        f"• Google Trends avg (7d; crypto/bitcoin/ethereum): "
        f"{color_flag(trends, t['trends_warn'], t['trends_flag'], True)} "
        f"{fmt_ratio(trends, 1)}  (warn ≥ {t['trends_warn']:.1f}, flag ≥ {t['trends_flag']:.1f})"
    )
    fng_today = m.get("fng_today")
    lines.append(
        f"• Fear & Greed Index (overall crypto): "
        f"{color_flag(fng_today, t['fng_warn'], t['fng_flag'], True)} "
        f"{fmt_ratio(fng_today, 0)}  (warn ≥ {t['fng_warn']:.0f}, flag ≥ {t['fng_flag']:.0f})"
    )
    fng14 = m.get("fng_ma14")
    lines.append(
        f"• Fear & Greed 14-day average: "
        f"{color_flag(fng14, t['fng14_warn'], t['fng14_flag'], True)} "
        f"{fmt_ratio(fng14, 1)}  (warn ≥ {t['fng14_warn']:.0f}, flag ≥ {t['fng14_flag']:.0f})"
    )
    fng30 = m.get("fng_ma30")
    lines.append(
        f"• Fear & Greed 30-day average: "
        f"{color_flag(fng30, t['fng30_warn'], t['fng30_flag'], True)} "
        f"{fmt_ratio(fng30, 1)}  (warn ≥ {t['fng30_warn']:.0f}, flag ≥ {t['fng30_flag']:.0f})"
    )
    drow = m.get("fng_persist_days")
    ppct = m.get("fng_persist_pct30")
    if drow is not None or ppct is not None:
        days_txt = str(drow) if drow is not None else "n/a"
        pct_txt = f"{ppct:.0f}%" if isinstance(ppct, (int, float)) else "n/a"
        # persistence coloring: max of days and pct scores
        days_score = linear_score(float(drow or 0), float(
            t["fng_days_warn"]), float(t["fng_days_flag"]), True)
        pct_score = linear_score(float(ppct or 0), float(
            t["fng_pct30_warn"]), float(t["fng_pct30_flag"]), True)
        col = "🔴" if max(days_score, pct_score) >= 100.0 else (
            "🟡" if max(days_score, pct_score) > 0 else "🟢")
        lines.append(
            f"• Greed persistence: {col} {days_txt} days in a row | {pct_txt} of last 30 days ≥ 70  "
            f"(warn: days ≥ {t['fng_days_warn']} or pct ≥ {t['fng_pct30_warn']:.0f}%; "
            f"flag: days ≥ {t['fng_days_flag']} or pct ≥ {t['fng_pct30_flag']:.0f}%)"
        )
    lines.append("")

    # Cycle & On-Chain
    lines.append("Cycle & On-Chain")
    pip = m.get("pi_prox")
    lines.append(
        f"• Pi Cycle Top proximity: "
        f"{color_flag(pip, t['pi_close_warn'], t['pi_close_flag'], False)} "
        f"{fmt_pct(pip, 2)} of trigger (100% = cross distance denominator)"
    )
    lines.append("")

    # Momentum (2W) & Extensions (1W)
    lines.append("Momentum (2W) & Extensions (1W)")
    btc_r = m.get("btc_rsi2w")
    btc_r_ma = m.get("btc_rsi2w_ma")
    lines.append(
        f"• BTC RSI (2W): {color_flag(btc_r, t['rsi_btc2w_warn'], t['rsi_btc2w_flag'], True)} "
        f"{fmt_ratio(btc_r, 1)} (MA {fmt_ratio(btc_r_ma, 1)}) (warn ≥ {t['rsi_btc2w_warn']:.1f}, flag ≥ {t['rsi_btc2w_flag']:.1f})"
    )
    ebr = m.get("ethbtc_rsi2w")
    ebr_ma = m.get("ethbtc_rsi2w_ma")
    lines.append(
        f"• ETH/BTC RSI (2W): {color_flag(ebr, t['rsi_ethbtc2w_warn'], t['rsi_ethbtc2w_flag'], True)} "
        f"{fmt_ratio(ebr, 1)} (MA {fmt_ratio(ebr_ma, 1)}) (warn ≥ {t['rsi_ethbtc2w_warn']:.1f}, flag ≥ {t['rsi_ethbtc2w_flag']:.1f})"
    )
    ar = m.get("alt_rsi2w")
    lines.append(
        f"• ALT basket (equal-weight) RSI (2W): {color_flag(ar, t['rsi_alt2w_warn'], t['rsi_alt2w_flag'], True)} "
        f"{fmt_ratio(ar, 1)} (warn ≥ {t['rsi_alt2w_warn']:.1f}, flag ≥ {t['rsi_alt2w_flag']:.1f})"
    )
    kb = m.get("btc_stoch2w_k")
    db = m.get("btc_stoch2w_d")
    ka = m.get("alt_stoch2w_k")
    da = m.get("alt_stoch2w_d")
    if kb is not None and db is not None:
        lines.append(
            f"• BTC Stoch RSI (2W) K/D: {fmt_ratio(kb, 2)}/{fmt_ratio(db, 2)} (overbought ≥ {t['stoch_ob']:.2f}; red = bearish cross from OB)")
    if ka is not None and da is not None:
        lines.append(
            f"• ALT basket Stoch RSI (2W) K/D: {fmt_ratio(ka, 2)}/{fmt_ratio(da, 2)} (overbought ≥ {t['stoch_ob']:.2f}; red = bearish cross from OB)")
    fb = m.get("btc_fib1272_prox")
    fa = m.get("alt_fib1272_prox")
    lines.append(
        f"• BTC Fibonacci extension proximity: "
        f"{color_flag(fb, t['fib_warn_pct'], t['fib_flag_pct'], False)} 1.272 @ {fmt_ratio(fb, 2)}% away "
        f"(warn ≤ {t['fib_warn_pct']:.1f}%, flag ≤ {t['fib_flag_pct']:.1f}%)"
    )
    lines.append(
        f"• ALT basket Fibonacci proximity: "
        f"{color_flag(fa, t['fib_warn_pct'], t['fib_flag_pct'], False)} 1.272 @ {fmt_ratio(fa, 2)}% away "
        f"(warn ≤ {t['fib_warn_pct']:.1f}%, flag ≤ {t['fib_flag_pct']:.1f}%)"
    )
    lines.append("")

    # Composite
    certainty, top5 = composite_certainty(m, t)
    cert_color = "🟢" if certainty < 40 else ("🟡" if certainty < 70 else "🔴")
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {cert_color} {certainty}/100 (yellow ≥ 40, red ≥ 70)")
    if top5:
        lines.append("• Top drivers:")
        for label, impact in top5:
            # map impact (weighted) into rough traffic-light:
            col = "🔴" if impact >= 66 else ("🟡" if impact >= 33 else "🟢")
            lines.append(f"  • {label}: {col} {int(round(impact))}/100")
    return "\n".join(lines)

# ---------- Telegram handlers ----------


async def cmd_start(update, context: ContextTypes.DEFAULT_TYPE):
    global LATEST_CHAT_ID
    LATEST_CHAT_ID = update.effective_chat.id
    msg = (
        "Hey! I’ll monitor cycle risk across market structure, derivatives, sentiment, cycle signals, and momentum.\n\n"
        "Commands:\n"
        "• /assess [conservative|moderate|aggressive] — on-demand snapshot (default: moderate)\n"
        "• /status — same as assess\n"
        "• /help — this help\n\n"
        "You’ll also get:\n"
        "• A daily summary (UTC)\n"
        "• Periodic alerts if any red flags trigger"
    )
    await update.message.reply_text(msg)


async def cmd_help(update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_start(update, context)


async def cmd_assess(update, context: ContextTypes.DEFAULT_TYPE):
    global LATEST_CHAT_ID
    LATEST_CHAT_ID = update.effective_chat.id
    profile = "moderate"
    if context.args:
        cand = context.args[0].lower().strip()
        if cand in {"conservative", "moderate", "aggressive"}:
            profile = cand
    await build_and_send_snapshot(context.application, update.effective_chat.id, profile)


async def cmd_status(update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_assess(update, context)


async def build_and_send_snapshot(app: Application, chat_id: int, profile: str = "moderate"):
    try:
        m = await gather_metrics(profile)
        text = build_snapshot_text(m, profile)
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        LOG.exception("build_and_send_snapshot failed")
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Could not fetch metrics right now: {e}")

# ---------- Scheduler jobs ----------


async def push_summary(app: Application):
    chat_id_env = os.getenv("DEFAULT_CHAT_ID")
    chat_id = int(chat_id_env) if chat_id_env and chat_id_env.isdigit() else (
        LATEST_CHAT_ID or None)
    if not chat_id:
        return
    await build_and_send_snapshot(app, chat_id, "moderate")


async def push_alerts(app: Application):
    # Minimal example: reuse summary for now
    await push_summary(app)

# ---------- Web health server ----------


async def handle_root(request):
    return web.Response(text="OK")


async def handle_health(request):
    return web.json_response({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


async def start_web_app() -> web.AppRunner:
    app = web.Application()
    app.add_routes([web.get("/", handle_root),
                   web.get("/health", handle_health)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    LOG.info("Health server listening on :%d", port)
    return runner

# ---------- Main ----------


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")

    # Build Telegram application
    app = ApplicationBuilder().token(token).build()

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("assess", cmd_assess))
    app.add_handler(CommandHandler("status", cmd_status))

    # Start web server
    runner = await start_web_app()

    # Start Telegram bot without taking over the event loop
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    LOG.info("Bot running. Press Ctrl+C to exit.")

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=os.getenv("TZ", "UTC"))
    # daily summary at 14:00 UTC
    scheduler.add_job(lambda: asyncio.create_task(
        push_summary(app)), "cron", hour=14, minute=0)
    # alerts every 15 minutes (same payload for now)
    scheduler.add_job(lambda: asyncio.create_task(
        push_alerts(app)), "cron", minute="*/15")
    scheduler.start()
    LOG.info("Scheduler started")

    try:
        # Run forever
        await asyncio.Event().wait()
    finally:
        scheduler.shutdown(wait=False)
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
