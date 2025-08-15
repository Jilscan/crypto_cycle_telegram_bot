# main.py
import os
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Set

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from aiohttp import web

import pandas as pd
import numpy as np
from pytrends.request import TrendReq

# ----------------------------
# Config & Globals
# ----------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("crypto-cycle-bot")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
SUMMARY_UTC_HOUR = int(os.getenv("SUMMARY_UTC_HOUR", "13"))
ALERT_CRON_MINUTES = os.getenv("ALERT_CRON_MINUTES", "*/15")
SHOW_TOP_DRIVERS = str(os.getenv("SHOW_TOP_DRIVERS", "0")
                       ).lower() in ("1", "true", "yes", "on")

THRESH = {
    "dominance_warn": 48.0, "dominance_flag": 60.0,
    "ethbtc_warn": 0.072, "ethbtc_flag": 0.090,
    "altcap_btc_warn": 1.44, "altcap_btc_flag": 1.80,
    "funding_warn": 0.08, "funding_flag": 0.10,  # percent
    "oi_btc_warn": 16e9, "oi_btc_flag": 20e9,
    "oi_eth_warn": 6.4e9, "oi_eth_flag": 8e9,
    "trends_warn": 60.0, "trends_flag": 75.0,
    "fng_warn": 56, "fng_flag": 70,
    "fng14_warn": 56, "fng14_flag": 70,
    "fng30_warn": 52, "fng30_flag": 65,
    "fng_persist_warn_days": 8, "fng_persist_flag_days": 10,
    "fng_persist_warn_pct30": 0.48, "fng_persist_flag_pct30": 0.60,
    "rsi_btc2w_warn": 60.0, "rsi_btc2w_flag": 70.0,
    "rsi_ethbtc2w_warn": 55.0, "rsi_ethbtc2w_flag": 65.0,
    "rsi_alt2w_warn": 65.0, "rsi_alt2w_flag": 75.0,
    "stoch_ob": 0.80,
    "fib_warn": 0.03, "fib_flag": 0.015
}

RISK_PROFILES = {
    "conservative": {"yellow": 35, "red": 60},
    "moderate":     {"yellow": 40, "red": 70},
    "aggressive":   {"yellow": 45, "red": 80},
}
DEFAULT_PROFILE = "moderate"

SUBSCRIBERS: Set[int] = set()
CHAT_PROFILE: Dict[int, str] = {}

OKX_FUNDING_INSTR = ["BTC-USDT-SWAP", "ETH-USDT-SWAP",
                     "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]
ALT_BASKET = ["SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
              "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"]

# ----------------------------
# Helpers (formatting & math)
# ----------------------------


def color_bullet(color: str) -> str:
    return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(color, "🟢")


def grade(value: Optional[float], warn: float, flag: float, higher_is_risk: bool = True) -> str:
    if value is None:
        return "yellow"
    if higher_is_risk:
        if value >= flag:
            return "red"
        if value >= warn:
            return "yellow"
        return "green"
    else:
        if value <= flag:
            return "red"
        if value <= warn:
            return "yellow"
        return "green"


def safe_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}%"


def fmt_pct(x: Optional[float], digits: int = 3) -> str:
    return f"{x:.{digits}f}%" if (x is not None and not (isinstance(x, float) and math.isnan(x))) else "n/a"


def safe_ratio(x: Optional[float], digits: int = 5) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def safe_num(x: Optional[float], digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def safe_big(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.3f}T"
    if absx >= 1e9:
        return f"${x/1e9:.3f}B"
    if absx >= 1e6:
        return f"${x/1e6:.3f}M"
    return f"${int(round(x)):,}"


def pct_away(current: float, target: float) -> float:
    if target == 0:
        return float("nan")
    return abs(current - target) / abs(target)


def stoch_rsi(series: pd.Series, period: int = 14, k: int = 3, d: int = 3) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(period, min_periods=period).min()
    max_rsi = rsi.rolling(period, min_periods=period).max()
    stoch = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-12)
    kline = stoch.rolling(k, min_periods=k).mean()
    dline = kline.rolling(d, min_periods=d).mean()
    return kline, dline


def to_2w(df_w: pd.DataFrame) -> pd.DataFrame:
    o = df_w['open'].resample('2W').first()
    h = df_w['high'].resample('2W').max()
    l = df_w['low'].resample('2W').min()
    c = df_w['close'].resample('2W').last()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()


def ohlc_from_okx_candles(rows: List[List[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    arr = []
    for r in rows:
        ts = int(r[0]) / 1000.0
        arr.append({
            "dt": datetime.fromtimestamp(ts, tz=timezone.utc),
            "open": float(r[1]), "high": float(r[2]),
            "low": float(r[3]), "close": float(r[4])
        })
    df = pd.DataFrame(arr).sort_values("dt").set_index("dt")
    return df

# ----------------------------
# Data Client
# ----------------------------


class DataClient:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def _get_json(self, url: str, params: Dict[str, Any] = None, retries: int = 3) -> Optional[Dict[str, Any]]:
        assert self.client is not None
        last_err = None
        for i in range(retries):
            try:
                r = await self.client.get(url, params=params)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                log.warning("GET %s failed (attempt %d): %s",
                            r.request.url if 'r' in locals() else url, i+1, e)
                await asyncio.sleep(0.8 * (i+1))
        log.warning("Failed to fetch %s", url)
        return None

    async def coingecko_global(self) -> Optional[Dict[str, Any]]:
        url = "https://api.coingecko.com/api/v3/global"
        return await self._get_json(url)

    async def coingecko_simple_price(self, ids: List[str]) -> Optional[Dict[str, Any]]:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": ",".join(ids), "vs_currencies": "usd"}
        return await self._get_json(url, params=params)

    async def okx_funding_rate(self, inst_id: str) -> Optional[float]:
        url = "https://www.okx.com/api/v5/public/funding-rate"
        js = await self._get_json(url, params={"instId": inst_id})
        try:
            val = float(js["data"][0]["fundingRate"])
            return val * 100.0
        except Exception:
            return None

    async def okx_ticker_last(self, inst_id: str) -> Optional[float]:
        url = "https://www.okx.com/api/v5/market/ticker"
        js = await self._get_json(url, params={"instId": inst_id})
        try:
            return float(js["data"][0]["last"])
        except Exception:
            return None

    async def okx_oi_usd(self, uly: str) -> Optional[float]:
        url = "https://www.okx.com/api/v5/public/open-interest"
        js = await self._get_json(url, params={"instType": "SWAP", "uly": uly})
        if not js or not js.get("data"):
            return None
        row = js["data"][0]
        oi_ccy = None
        try:
            oi_ccy = float(row.get("oiCcy")) if row.get(
                "oiCcy") not in ("", None) else None
        except Exception:
            oi_ccy = None
        last = await self.okx_ticker_last(uly + "-SWAP")
        if oi_ccy is not None and last is not None:
            return oi_ccy * last
        try:
            return float(row.get("oiUsd"))
        except Exception:
            return None

    async def okx_candles(self, inst_id: str, bar: str, limit: int) -> Optional[pd.DataFrame]:
        url = "https://www.okx.com/api/v5/market/candles"
        js = await self._get_json(url, params={"instId": inst_id, "bar": bar, "limit": str(limit)})
        try:
            rows = js["data"]
            return ohlc_from_okx_candles(rows)
        except Exception:
            return None

    async def google_trends_avg(self, queries: List[str], days: int = 7) -> Optional[float]:
        def _do() -> Optional[float]:
            try:
                py = TrendReq(hl="en-US", tz=0)
                py.build_payload(
                    kw_list=queries, timeframe=f"now {days}-d", geo="")
                df = py.interest_over_time()
                if df is None or df.empty:
                    return None
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                return float(df.mean().mean())
            except Exception as e:
                log.warning("google trends failed: %s", e)
                return None
        return await asyncio.to_thread(_do)

    async def fear_greed_latest(self) -> Optional[int]:
        js = await self._get_json("https://api.alternative.me/fng/", params={"limit": "1"})
        try:
            return int(js["data"][0]["value"])
        except Exception:
            return None

    async def fear_greed_series(self, limit: int = 60) -> Optional[List[int]]:
        js = await self._get_json("https://api.alternative.me/fng/", params={"limit": str(limit)})
        try:
            return [int(x["value"]) for x in reversed(js["data"])]
        except Exception:
            return None

# ----------------------------
# Metric computation
# ----------------------------


async def fetch_metrics(profile: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"_ts": datetime.now(
        timezone.utc).isoformat(), "_profile": profile}

    async with DataClient() as dc:
        cg = await dc.coingecko_global()
        total_mcap = btc_mcap = dominance = None
        if cg and cg.get("data"):
            dd = cg["data"]
            try:
                total_mcap = float(dd["total_market_cap"]["usd"])
                dominance = float(dd["market_cap_percentage"]["btc"])
                btc_mcap = total_mcap * (dominance / 100.0)
            except Exception:
                pass
        out["btc_dominance_pct"] = dominance

        sp = await dc.coingecko_simple_price(["bitcoin", "ethereum"])
        btc_price = eth_price = None
        if sp:
            try:
                btc_price = float(sp["bitcoin"]["usd"])
                eth_price = float(sp["ethereum"]["usd"])
            except Exception:
                pass
        out["eth_btc"] = (
            eth_price / btc_price) if (eth_price and btc_price) else None

        altcap_btc_ratio = None
        if total_mcap and btc_mcap and btc_mcap > 0:
            altcap_btc_ratio = max((total_mcap - btc_mcap), 0.0) / btc_mcap
        out["altcap_btc_ratio"] = altcap_btc_ratio

        funding_vals: List[Tuple[str, Optional[float]]] = []
        for inst in OKX_FUNDING_INSTR:
            val = await dc.okx_funding_rate(inst)
            funding_vals.append((inst.replace("-USDT-SWAP", "USDT"), val))
        valid_f = [v for _, v in funding_vals if v is not None]
        out["funding_top3"] = sorted(funding_vals, key=lambda x: (
            abs(x[1]) if x[1] is not None else -1), reverse=True)[:3]
        out["funding_max_abs"] = max([abs(v) for v in valid_f], default=None)
        out["funding_median_abs"] = float(
            np.median([abs(v) for v in valid_f])) if valid_f else None

        out["oi_btc_usd"] = await dc.okx_oi_usd("BTC-USDT")
        out["oi_eth_usd"] = await dc.okx_oi_usd("ETH-USDT")

        out["trends_avg7"] = await dc.google_trends_avg(["crypto", "bitcoin", "ethereum"], days=7)

        fng_today = await dc.fear_greed_latest()
        fng_series = await dc.fear_greed_series(60)
        out["fng_today"] = fng_today
        if fng_series:
            s = pd.Series(fng_series, dtype=float)
            out["fng_ma14"] = float(
                s.tail(14).mean()) if len(s) >= 14 else None
            out["fng_ma30"] = float(
                s.tail(30).mean()) if len(s) >= 30 else None
            consec = 0
            for v in reversed(fng_series):
                if v >= 70:
                    consec += 1
                else:
                    break
            out["fng_greed_streak"] = consec
            last30 = fng_series[-30:] if len(fng_series) >= 30 else fng_series
            out["fng_greed_pct30"] = sum(
                1 for v in last30 if v >= 70) / len(last30)
        else:
            out["fng_ma14"] = None
            out["fng_ma30"] = None
            out["fng_greed_streak"] = None
            out["fng_greed_pct30"] = None

        btc_w = await dc.okx_candles("BTC-USDT", "1W", 400)
        btc_d = await dc.okx_candles("BTC-USDT", "1D", 500)
        ethbtc_w = await dc.okx_candles("ETH-BTC", "1W", 400)

        basket_w: Dict[str, pd.DataFrame] = {}
        for sym in ALT_BASKET:
            dfw = await dc.okx_candles(sym, "1W", 400)
            if dfw is not None and not dfw.empty:
                basket_w[sym] = dfw

        def rsi_2w(dfw: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
            if dfw is None or dfw.empty:
                return None, None
            dfw2 = to_2w(dfw)
            if dfw2.empty:
                return None, None
            close = dfw2["close"]
            delta = close.diff()
            up = delta.clip(lower=0.0)
            down = -delta.clip(upper=0.0)
            roll_up = up.ewm(alpha=1/14, adjust=False).mean()
            roll_down = down.ewm(alpha=1/14, adjust=False).mean()
            rs = roll_up / (roll_down + 1e-12)
            rsi = 100 - (100 / (1 + rs))
            rsi_ma = rsi.rolling(10, min_periods=5).mean()
            return float(rsi.iloc[-1]), float(rsi_ma.iloc[-1]) if not math.isnan(rsi_ma.iloc[-1]) else None

        btc_rsi2w, btc_rsi2w_ma = rsi_2w(btc_w)
        ethbtc_rsi2w, _ = rsi_2w(ethbtc_w)

        def stoch2w(dfw: Optional[pd.DataFrame]) -> Tuple[Optional[float], Optional[float]]:
            if dfw is None or dfw.empty:
                return None, None
            dfw2 = to_2w(dfw)
            if dfw2.empty:
                return None, None
            close = pd.Series(dfw2["close"])
            k, d = stoch_rsi(close, period=14, k=3, d=3)
            if k.empty or d.empty or math.isnan(k.iloc[-1]) or math.isnan(d.iloc[-1]):
                return None, None
            return float(k.iloc[-1]), float(d.iloc[-1])

        btc_k2w, btc_d2w = stoch2w(btc_w)
        ethbtc_k2w, ethbtc_d2w = stoch2w(ethbtc_w)

        alt_rsi2w = alt_k2w = alt_d2w = None
        if basket_w:
            aligned = pd.DataFrame(
                {k: v["close"] for k, v in basket_w.items()}).dropna(how="any")
            if not aligned.empty:
                norm = aligned / aligned.iloc[0] * 100.0
                dfw_idx = pd.DataFrame({"close": norm.mean(axis=1)})
                dfw_idx["open"] = dfw_idx["close"]
                dfw_idx["high"] = dfw_idx["close"]
                dfw_idx["low"] = dfw_idx["close"]
                dfw_idx = dfw_idx[["open", "high", "low", "close"]]
                dfw_idx.index = pd.to_datetime(dfw_idx.index)
                alt_rsi2w, _ = rsi_2w(dfw_idx)
                alt_k2w, alt_d2w = stoch2w(dfw_idx)

        out["btc_rsi2w"] = btc_rsi2w
        out["btc_rsi2w_ma"] = btc_rsi2w_ma
        out["ethbtc_rsi2w"] = ethbtc_rsi2w
        out["btc_k2w"] = btc_k2w
        out["btc_d2w"] = btc_d2w
        out["ethbtc_k2w"] = ethbtc_k2w
        out["ethbtc_d2w"] = ethbtc_d2w
        out["alt_rsi2w"] = alt_rsi2w
        out["alt_k2w"] = alt_k2w
        out["alt_d2w"] = alt_d2w

        def fib_1272_proximity(dfw_close: Optional[pd.Series]) -> Optional[Tuple[float, float]]:
            if dfw_close is None or dfw_close.empty:
                return None
            c = dfw_close.tail(100)
            if len(c) < 10:
                return None
            lo = float(c.min())
            hi = float(c.max())
            if hi <= lo:
                return None
            ext_1272 = hi + 0.272 * (hi - lo)
            cur = float(c.iloc[-1])
            away = pct_away(cur, ext_1272)
            return ext_1272, away

        if btc_w is not None and not btc_w.empty:
            btc_ext = fib_1272_proximity(btc_w["close"])
            if btc_ext:
                out["btc_fib1272_price"] = btc_ext[0]
                out["btc_fib1272_away"] = btc_ext[1]
            else:
                out["btc_fib1272_price"] = None
                out["btc_fib1272_away"] = None
        else:
            out["btc_fib1272_price"] = None
            out["btc_fib1272_away"] = None

        if basket_w:
            aligned = pd.DataFrame(
                {k: v["close"] for k, v in basket_w.items()}).dropna(how="any")
            if not aligned.empty:
                idx = aligned / aligned.iloc[0] * 100.0
                idx_close = idx.mean(axis=1)
                alt_ext = fib_1272_proximity(idx_close)
                if alt_ext:
                    out["alt_fib1272_price"] = alt_ext[0]
                    out["alt_fib1272_away"] = alt_ext[1]
                else:
                    out["alt_fib1272_price"] = None
                    out["alt_fib1272_away"] = None
            else:
                out["alt_fib1272_price"] = None
                out["alt_fib1272_away"] = None
        else:
            out["alt_fib1272_price"] = None
            out["alt_fib1272_away"] = None

        def pi_cycle_proximity(df_daily: Optional[pd.DataFrame]) -> Optional[float]:
            if df_daily is None or df_daily.empty:
                return None
            c = df_daily["close"]
            if len(c) < 400:
                return None
            sma111 = c.rolling(111, min_periods=111).mean()
            sma350 = c.rolling(350, min_periods=350).mean()
            a = sma111.iloc[-1]
            b = 2.0 * sma350.iloc[-1]
            if math.isnan(a) or math.isnan(b) or b == 0:
                return None
            prox = max(0.0, 1.0 - abs(a - b) / abs(b)) * 100.0
            return prox

        out["pi_proximity_pct"] = pi_cycle_proximity(btc_d)

    return out

# ----------------------------
# Composite Scoring
# ----------------------------


def composite_score(m: Dict[str, Any], profile: str) -> Tuple[float, List[Tuple[str, float, float, str]]]:
    weights = {
        "market": 0.20,
        "deriv":  0.20,
        "sent":   0.25,
        "moment": 0.25,
        "fibpi":  0.10
    }

    drivers: List[Tuple[str, float, float, str]] = []

    def ramp(v: Optional[float], warn: float, flag: float, higher_is_risk: bool = True) -> float:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 50.0
        if higher_is_risk:
            if v <= warn:
                return 0.0
            if v >= flag:
                return 100.0
            return 100.0 * (v - warn) / (flag - warn)
        else:
            if v >= warn:
                return 0.0
            if v <= flag:
                return 100.0
            return 100.0 * (warn - v) / (warn - flag)

    d_sc = ramp(m.get("btc_dominance_pct"),
                THRESH["dominance_warn"], THRESH["dominance_flag"], True)
    e_sc = ramp(m.get("eth_btc"),
                THRESH["ethbtc_warn"], THRESH["ethbtc_flag"], True)
    a_sc = ramp(m.get("altcap_btc_ratio"),
                THRESH["altcap_btc_warn"], THRESH["altcap_btc_flag"], True)
    mkt_score = (d_sc + e_sc + a_sc) / 3.0
    drivers += [("BTC dominance", d_sc, weights["market"]/3, "red" if d_sc >= 70 else "yellow" if d_sc >= 35 else "green"),
                ("ETH/BTC", e_sc, weights["market"]/3, "red" if e_sc >=
                 70 else "yellow" if e_sc >= 35 else "green"),
                ("AltCap/BTC", a_sc, weights["market"]/3, "red" if a_sc >= 70 else "yellow" if a_sc >= 35 else "green")]

    f_sc = ramp(m.get("funding_max_abs"),
                THRESH["funding_warn"], THRESH["funding_flag"], True)
    oi_btc_sc = ramp(m.get("oi_btc_usd"),
                     THRESH["oi_btc_warn"], THRESH["oi_btc_flag"], True)
    oi_eth_sc = ramp(m.get("oi_eth_usd"),
                     THRESH["oi_eth_warn"], THRESH["oi_eth_flag"], True)
    deriv_score = (f_sc + oi_btc_sc + oi_eth_sc) / 3.0
    drivers += [("Funding", f_sc, weights["deriv"]/3, "red" if f_sc >= 70 else "yellow" if f_sc >= 35 else "green"),
                ("OI BTC", oi_btc_sc, weights["deriv"]/3, "red" if oi_btc_sc >=
                 70 else "yellow" if oi_btc_sc >= 35 else "green"),
                ("OI ETH", oi_eth_sc, weights["deriv"]/3, "red" if oi_eth_sc >= 70 else "yellow" if oi_eth_sc >= 35 else "green")]

    tr_sc = ramp(m.get("trends_avg7"),
                 THRESH["trends_warn"], THRESH["trends_flag"], True)
    f_sc0 = ramp(m.get("fng_today"),
                 THRESH["fng_warn"], THRESH["fng_flag"], True)
    f_sc14 = ramp(m.get("fng_ma14"),
                  THRESH["fng14_warn"], THRESH["fng14_flag"], True)
    f_sc30 = ramp(m.get("fng_ma30"),
                  THRESH["fng30_warn"], THRESH["fng30_flag"], True)
    days = m.get("fng_greed_streak")
    pct30 = m.get("fng_greed_pct30")
    pers = 0.0
    if isinstance(days, int):
        pers = max(pers, ramp(float(
            days), THRESH["fng_persist_warn_days"], THRESH["fng_persist_flag_days"], True))
    if isinstance(pct30, float):
        pers = max(pers, ramp(
            pct30, THRESH["fng_persist_warn_pct30"], THRESH["fng_persist_flag_pct30"], True))
    sent_score = np.nanmean([tr_sc, f_sc0, f_sc14, f_sc30, pers])
    drivers += [("Trends", tr_sc, weights["sent"]/5, "red" if tr_sc >= 70 else "yellow" if tr_sc >= 35 else "green"),
                ("F&G today", f_sc0, weights["sent"]/5, "red" if f_sc0 >=
                 70 else "yellow" if f_sc0 >= 35 else "green"),
                ("F&G 14d", f_sc14, weights["sent"]/5, "red" if f_sc14 >=
                 70 else "yellow" if f_sc14 >= 35 else "green"),
                ("F&G 30d", f_sc30, weights["sent"]/5, "red" if f_sc30 >=
                 70 else "yellow" if f_sc30 >= 35 else "green"),
                ("F&G persist", pers, weights["sent"]/5, "red" if pers >= 70 else "yellow" if pers >= 35 else "green")]

    rsi_btc = ramp(m.get("btc_rsi2w"),
                   THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"], True)
    rsi_ethbtc = ramp(m.get(
        "ethbtc_rsi2w"), THRESH["rsi_ethbtc2w_warn"], THRESH["rsi_ethbtc2w_flag"], True)
    rsi_alt = ramp(m.get("alt_rsi2w"),
                   THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"], True)

    stoch_btc = 0.0
    if m.get("btc_k2w") is not None and m.get("btc_d2w") is not None:
        k, d = m["btc_k2w"], m["btc_d2w"]
        if k >= THRESH["stoch_ob"]:
            stoch_btc = 60.0 if k <= d else 80.0
    stoch_alt = 0.0
    if m.get("alt_k2w") is not None and m.get("alt_d2w") is not None:
        k, d = m["alt_k2w"], m["alt_d2w"]
        if k >= THRESH["stoch_ob"]:
            stoch_alt = 60.0 if k <= d else 80.0

    if m.get("btc_rsi2w") is not None and m.get("btc_rsi2w_ma") is not None and m["btc_rsi2w"] < m["btc_rsi2w_ma"]:
        rsi_btc = min(100.0, rsi_btc + 10.0)

    mom_score = np.nanmean(
        [rsi_btc, rsi_ethbtc, rsi_alt, stoch_btc, stoch_alt])
    drivers += [("RSI BTC 2W", rsi_btc, weights["moment"]/5, "red" if rsi_btc >= 70 else "yellow" if rsi_btc >= 35 else "green"),
                ("RSI ETHBTC 2W", rsi_ethbtc,
                 weights["moment"]/5, "red" if rsi_ethbtc >= 70 else "yellow" if rsi_ethbtc >= 35 else "green"),
                ("RSI ALT 2W", rsi_alt, weights["moment"]/5, "red" if rsi_alt >=
                 70 else "yellow" if rsi_alt >= 35 else "green"),
                ("Stoch BTC 2W", stoch_btc,
                 weights["moment"]/5, "red" if stoch_btc >= 70 else "yellow" if stoch_btc >= 35 else "green"),
                ("Stoch ALT 2W", stoch_alt, weights["moment"]/5, "red" if stoch_alt >= 70 else "yellow" if stoch_alt >= 35 else "green")]

    fib_btc = 0.0
    if m.get("btc_fib1272_away") is not None:
        away = m["btc_fib1272_away"]
        if away <= THRESH["fib_flag"]:
            fib_btc = 100.0
        elif away <= THRESH["fib_warn"]:
            fib_btc = 60.0
        else:
            fib_btc = max(0.0, 100.0 * (THRESH["fib_warn"]/away - 0.5))

    pi = 0.0
    if m.get("pi_proximity_pct") is not None:
        v = m["pi_proximity_pct"]
        if v >= 90:
            pi = 100.0
        elif v >= 60:
            pi = 70.0
        elif v >= 40:
            pi = 40.0
        else:
            pi = 10.0

    fibpi_score = np.nanmean([fib_btc, pi])
    drivers += [("Fib 1.272 BTC", fib_btc, weights["fibpi"]/2, "red" if fib_btc >= 70 else "yellow" if fib_btc >= 35 else "green"),
                ("Pi proximity",  pi,      weights["fibpi"]/2, "red" if pi >= 70 else "yellow" if pi >= 35 else "green")]

    comp = (mkt_score * weights["market"] +
            deriv_score * weights["deriv"] +
            sent_score * weights["sent"] +
            mom_score * weights["moment"] +
            fibpi_score * weights["fibpi"])
    comp = max(0.0, min(100.0, float(comp)))
    return comp, drivers

# ----------------------------
# Message Builder
# ----------------------------


def build_message(m: Dict[str, Any], profile: str) -> str:
    ts = m.get("_ts", datetime.now(timezone.utc).isoformat())
    dt_short = ts.split(".")[0].replace("+00:00", "Z")
    lines: List[str] = []
    lines.append(f"📊 Crypto Market Snapshot — {dt_short}")
    lines.append(f"Profile: {profile}")
    lines.append("")

    # Market Structure
    lines.append("Market Structure")
    dom = m.get("btc_dominance_pct")
    ethbtc = m.get("eth_btc")
    acr = m.get("altcap_btc_ratio")
    c_dom = grade(dom, THRESH["dominance_warn"],
                  THRESH["dominance_flag"], True)
    c_eth = grade(ethbtc, THRESH["ethbtc_warn"], THRESH["ethbtc_flag"], True)
    c_acr = grade(acr, THRESH["altcap_btc_warn"],
                  THRESH["altcap_btc_flag"], True)
    lines.append(
        f"• Bitcoin market share of total crypto: {color_bullet(c_dom)} "
        f"{safe_num(dom,2)}%  (warn ≥ {THRESH['dominance_warn']:.2f}%, flag ≥ {THRESH['dominance_flag']:.2f}%)"
    )
    lines.append(
        f"• Ether price relative to Bitcoin (ETH/BTC): {color_bullet(c_eth)} "
        f"{safe_ratio(ethbtc,5)}  (warn ≥ {THRESH['ethbtc_warn']:.5f}, flag ≥ {THRESH['ethbtc_flag']:.5f})"
    )
    lines.append(
        f"• Altcoin market cap / Bitcoin market cap: {color_bullet(c_acr)} "
        f"{safe_num(acr,2)}  (warn ≥ {THRESH['altcap_btc_warn']:.2f}, flag ≥ {THRESH['altcap_btc_flag']:.2f})"
    )
    lines.append("")

    # Derivatives
    lines.append("Derivatives")
    fmax = m.get("funding_max_abs")
    fmed = m.get("funding_median_abs")
    c_f = grade(fmax, THRESH["funding_warn"], THRESH["funding_flag"], True)
    top3 = m.get("funding_top3", [])
    top3_txt = ", ".join(
        [f"{sym} {fmt_pct(val,3)}" for sym, val in top3]) if top3 else "n/a"
    lines.append(
        f"• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — max: "
        f"{color_bullet(c_f)} {fmt_pct(fmax,3)} | median: "
        f"{color_bullet(grade(fmed, THRESH['funding_warn'], THRESH['funding_flag'], True))} {fmt_pct(fmed,3)}  "
        f"(warn ≥ {THRESH['funding_warn']:.3f}%, flag ≥ {THRESH['funding_flag']:.3f}%)"
    )
    lines.append(f"  Top-3 funding extremes: {top3_txt}")

    oi_btc = m.get("oi_btc_usd")
    oi_eth = m.get("oi_eth_usd")
    c_oi_b = grade(oi_btc, THRESH["oi_btc_warn"], THRESH["oi_btc_flag"], True)
    c_oi_e = grade(oi_eth, THRESH["oi_eth_warn"], THRESH["oi_eth_flag"], True)
    lines.append(
        f"• Bitcoin open interest (USD): {color_bullet(c_oi_b)} {safe_big(oi_btc)}  "
        f"(warn ≥ {safe_big(THRESH['oi_btc_warn'])}, flag ≥ {safe_big(THRESH['oi_btc_flag'])})"
    )
    lines.append(
        f"• Ether open interest (USD): {color_bullet(c_oi_e)} {safe_big(oi_eth)}  "
        f"(warn ≥ {safe_big(THRESH['oi_eth_warn'])}, flag ≥ {safe_big(THRESH['oi_eth_flag'])})"
    )
    lines.append("")

    # Sentiment
    lines.append("Sentiment")
    tr = m.get("trends_avg7")
    c_tr = grade(tr, THRESH["trends_warn"], THRESH["trends_flag"], True)
    lines.append(
        f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {color_bullet(c_tr)} "
        f"{safe_num(tr,1)}  (warn ≥ {THRESH['trends_warn']:.1f}, flag ≥ {THRESH['trends_flag']:.1f})"
    )
    f0 = m.get("fng_today")
    c_f0 = grade(f0, THRESH["fng_warn"], THRESH["fng_flag"], True)
    f14 = m.get("fng_ma14")
    c_f14 = grade(f14, THRESH["fng14_warn"], THRESH["fng14_flag"], True)
    f30 = m.get("fng_ma30")
    c_f30 = grade(f30, THRESH["fng30_warn"], THRESH["fng30_flag"], True)
    streak = m.get("fng_greed_streak")
    pct30 = m.get("fng_greed_pct30")

    pers_score = 0.0
    if isinstance(streak, int):
        pers_score = max(pers_score, 100.0 if streak >= THRESH["fng_persist_flag_days"] else (
            60.0 if streak >= THRESH["fng_persist_warn_days"] else 0.0))
    if isinstance(pct30, float):
        ps = 100.0 if pct30 >= THRESH["fng_persist_flag_pct30"] else (
            60.0 if pct30 >= THRESH["fng_persist_warn_pct30"] else 0.0)
        pers_score = max(pers_score, ps)
    c_pers = "red" if pers_score >= 70 else "yellow" if pers_score >= 35 else "green"

    lines.append(
        f"• Fear & Greed Index (overall crypto): {color_bullet(c_f0)} "
        f"{str(f0) if f0 is not None else 'n/a'}  (warn ≥ {THRESH['fng_warn']}, flag ≥ {THRESH['fng_flag']})"
    )
    lines.append(
        f"• Fear & Greed 14-day average: {color_bullet(c_f14)} "
        f"{safe_num(f14,1)}  (warn ≥ {THRESH['fng14_warn']}, flag ≥ {THRESH['fng14_flag']})"
    )
    lines.append(
        f"• Fear & Greed 30-day average: {color_bullet(c_f30)} "
        f"{safe_num(f30,1)}  (warn ≥ {THRESH['fng30_warn']}, flag ≥ {THRESH['fng30_flag']})"
    )
    lines.append(
        f"• Greed persistence: {color_bullet(c_pers)} "
        f"{(str(streak) if streak is not None else 'n/a')} days in a row | "
        f"{(f'{pct30*100:.0f}%' if isinstance(pct30, float) else 'n/a')} of last 30 days ≥ 70  "
        f"(warn: days ≥ {THRESH['fng_persist_warn_days']} or pct ≥ {int(THRESH['fng_persist_warn_pct30']*100)}%; "
        f"flag: days ≥ {THRESH['fng_persist_flag_days']} or pct ≥ {int(THRESH['fng_persist_flag_pct30']*100)}%)"
    )
    lines.append("")

    # Cycle & On-Chain
    lines.append("Cycle & On-Chain")
    pi = m.get("pi_proximity_pct")
    c_pi = "red" if (isinstance(pi, float) and pi >= 90) else "yellow" if (
        isinstance(pi, float) and pi >= 60) else "green"
    lines.append(
        f"• Pi Cycle Top proximity: {color_bullet(c_pi)} "
        f"{safe_num(pi,2)}% of trigger (100% = cross)"
    )
    lines.append("")

    # Momentum (2W) & Extensions (1W)
    lines.append("Momentum (2W) & Extensions (1W)")
    c_rb = grade(m.get("btc_rsi2w"),
                 THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"], True)
    if m.get("btc_rsi2w") is not None:
        if m.get("btc_rsi2w_ma") is not None:
            lines.append(
                f"• BTC RSI (2W): {color_bullet(c_rb)} {m['btc_rsi2w']:.1f} (MA {m['btc_rsi2w_ma']:.1f}) "
                f"(warn ≥ {THRESH['rsi_btc2w_warn']:.1f}, flag ≥ {THRESH['rsi_btc2w_flag']:.1f})"
            )
        else:
            lines.append(
                f"• BTC RSI (2W): {color_bullet(c_rb)} {m['btc_rsi2w']:.1f} "
                f"(warn ≥ {THRESH['rsi_btc2w_warn']:.1f}, flag ≥ {THRESH['rsi_btc2w_flag']:.1f})"
            )
    else:
        lines.append("• BTC RSI (2W): 🟡 n/a")

    c_reb = grade(m.get("ethbtc_rsi2w"),
                  THRESH["rsi_ethbtc2w_warn"], THRESH["rsi_ethbtc2w_flag"], True)
    ethbtc_rsi_txt = safe_num(m.get("ethbtc_rsi2w"), 1)
    lines.append(
        f"• ETH/BTC RSI (2W): {color_bullet(c_reb)} {ethbtc_rsi_txt} "
        f"(warn ≥ {THRESH['rsi_ethbtc2w_warn']:.1f}, flag ≥ {THRESH['rsi_ethbtc2w_flag']:.1f})"
    )

    if m.get("btc_k2w") is not None and m.get("btc_d2w") is not None:
        lines.append(
            f"• BTC Stoch RSI (2W) K/D: {m['btc_k2w']:.2f}/{m['btc_d2w']:.2f} (overbought ≥ {THRESH['stoch_ob']:.2f})")
    else:
        lines.append("• BTC Stoch RSI (2W) K/D: n/a")

    c_ralt = grade(m.get("alt_rsi2w"),
                   THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"], True)
    alt_rsi_txt = safe_num(m.get("alt_rsi2w"), 1)
    lines.append(
        f"• ALT basket (equal-weight) RSI (2W): {color_bullet(c_ralt)} {alt_rsi_txt} "
        f"(warn ≥ {THRESH['rsi_alt2w_warn']:.1f}, flag ≥ {THRESH['rsi_alt2w_flag']:.1f})"
    )
    if m.get("alt_k2w") is not None and m.get("alt_d2w") is not None:
        lines.append(
            f"• ALT basket (equal-weight) Stoch RSI (2W) K/D: {m['alt_k2w']:.2f}/{m['alt_d2w']:.2f} (overbought ≥ {THRESH['stoch_ob']:.2f})")
    else:
        lines.append("• ALT basket (equal-weight) Stoch RSI (2W) K/D: n/a")

    fib_btc_away = m.get("btc_fib1272_away")
    c_fib_btc = "red" if (isinstance(fib_btc_away, float) and fib_btc_away <= THRESH["fib_flag"]) else \
                "yellow" if (isinstance(fib_btc_away, float)
                             and fib_btc_away <= THRESH["fib_warn"]) else "green"
    btc_fib_price_txt = safe_num(m.get("btc_fib1272_price"), 2)
    lines.append(
        f"• BTC Fibonacci extension proximity: {color_bullet(c_fib_btc)} "
        f"1.272 @ {btc_fib_price_txt} away {safe_pct((fib_btc_away*100 if isinstance(fib_btc_away, float) else None),2)} "
        f"(warn ≤ {THRESH['fib_warn']*100:.1f}%, flag ≤ {THRESH['fib_flag']*100:.1f}%)"
    )

    fib_alt_away = m.get("alt_fib1272_away")
    c_fib_alt = "red" if (isinstance(fib_alt_away, float) and fib_alt_away <= THRESH["fib_flag"]) else \
                "yellow" if (isinstance(fib_alt_away, float)
                             and fib_alt_away <= THRESH["fib_warn"]) else "green"
    alt_fib_price_txt = safe_num(m.get("alt_fib1272_price"), 2)
    lines.append(
        f"• ALT basket Fibonacci proximity: {color_bullet(c_fib_alt)} "
        f"1.272 @ {alt_fib_price_txt} away {safe_pct((fib_alt_away*100 if isinstance(fib_alt_away, float) else None),2)} "
        f"(warn ≤ {THRESH['fib_warn']*100:.1f}%, flag ≤ {THRESH['fib_flag']*100:.1f}%)"
    )
    lines.append("")

    # Composite
    comp, drivers = composite_score(m, profile)
    thr = RISK_PROFILES.get(profile, RISK_PROFILES[DEFAULT_PROFILE])
    yellow_cut = float(thr["yellow"])
    red_cut = float(thr["red"])
    comp_color = "green" if comp < yellow_cut else (
        "yellow" if comp < red_cut else "red")
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {color_bullet(comp_color)} {int(round(comp))}/100 (yellow ≥ {int(yellow_cut)}, red ≥ {int(red_cut)})")

    if SHOW_TOP_DRIVERS and drivers:
        ranked = sorted(drivers, key=lambda x: x[1]*x[2], reverse=True)[:5]
        lines.append("• Top drivers:")
        for name, s, w, c in ranked:
            lines.append(
                f"  • {name}: {color_bullet(c)} {int(round(s))}/100 (w={int(round(w*100))}%)")

    flags: List[str] = []
    if c_f0 == "red":
        flags.append("Greed is elevated (Fear & Greed Index)")
    if c_f30 == "red":
        flags.append("Fear & Greed 30-day avg in Greed")
    if c_pers == "red":
        flags.append("Greed persistent in last 30 days")
    if flags:
        lines.append("")
        lines.append(f"⚠️ Triggered flags ({len(flags)}): " + ", ".join(flags))

    return "\n".join(lines)

# ----------------------------
# Telegram Bot Handlers
# ----------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    if chat_id not in CHAT_PROFILE:
        CHAT_PROFILE[chat_id] = DEFAULT_PROFILE
    msg = (
        "Hi! I’ll send crypto cycle snapshots here.\n\n"
        "Commands:\n"
        "• /assess — get a fresh snapshot now\n"
        "• /setprofile <conservative|moderate|aggressive>\n"
        "• /subscribe — receive the daily summary + alerts\n"
        "• /unsubscribe — stop the automatic messages\n"
        "• /help — show this help"
    )
    await update.message.reply_text(msg)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, context)


async def cmd_setprofile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Usage: /setprofile conservative|moderate|aggressive")
        return
    choice = context.args[0].strip().lower()
    if choice not in RISK_PROFILES:
        await update.message.reply_text("Pick one of: conservative, moderate, aggressive")
        return
    CHAT_PROFILE[chat_id] = choice
    await update.message.reply_text(f"OK — profile set to {choice}.")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    SUBSCRIBERS.add(update.effective_chat.id)
    await update.message.reply_text("Subscribed ✅")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id in SUBSCRIBERS:
        SUBSCRIBERS.remove(chat_id)
    await update.message.reply_text("Unsubscribed ✅")


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    profile = CHAT_PROFILE.get(chat_id, DEFAULT_PROFILE)
    try:
        m = await fetch_metrics(profile)
        text = build_message(m, profile)
        await context.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        log.exception("assess failed")
        await update.message.reply_text(f"⚠️ Could not fetch metrics right now: {e}")

# ----------------------------
# Scheduled Jobs
# ----------------------------


async def push_summary(app: Application):
    if not SUBSCRIBERS:
        return
    for chat_id in list(SUBSCRIBERS):
        profile = CHAT_PROFILE.get(chat_id, DEFAULT_PROFILE)
        try:
            m = await fetch_metrics(profile)
            text = build_message(m, profile)
            await app.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            log.warning("push_summary to %s failed: %s", chat_id, e)


async def push_alerts(app: Application):
    if not SUBSCRIBERS:
        return
    for chat_id in list(SUBSCRIBERS):
        profile = CHAT_PROFILE.get(chat_id, DEFAULT_PROFILE)
        try:
            m = await fetch_metrics(profile)
            comp, _ = composite_score(m, profile)
            thr = RISK_PROFILES.get(profile, RISK_PROFILES[DEFAULT_PROFILE])
            if comp >= thr["red"]:
                await app.bot.send_message(chat_id=chat_id, text=f"🚨 Composite certainty {int(round(comp))}/100 (red). Consider de-risking.")
        except Exception as e:
            log.warning("push_alerts to %s failed: %s", chat_id, e)

# ----------------------------
# Health server (Koyeb)
# ----------------------------


async def health_handler(request):
    return web.json_response({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


async def start_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    log.info("Health server listening on :8080")

# ----------------------------
# Main
# ----------------------------


async def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")

    await start_health_server()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("setprofile", cmd_setprofile))
    application.add_handler(CommandHandler("subscribe", cmd_subscribe))
    application.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    application.add_handler(CommandHandler("assess", cmd_assess))

    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(lambda: asyncio.create_task(push_summary(application)),
                      trigger="cron", hour=SUMMARY_UTC_HOUR, minute=0)
    scheduler.add_job(lambda: asyncio.create_task(push_alerts(application)),
                      trigger="cron", minute=ALERT_CRON_MINUTES)
    scheduler.start()
    log.info("Scheduler started")

    log.info("Bot running. Press Ctrl+C to exit.")
    await application.initialize()
    await application.start()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await application.stop()
        await application.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
