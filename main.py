# main.py
import os
import asyncio
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import httpx
import pandas as pd
import numpy as np

from pytrends.request import TrendReq
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tzlocal import get_localzone
from aiohttp import web

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    ContextTypes
)

# ------------------------- Logging -------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("crypto-cycle-bot")

# ------------------------- Config --------------------------
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_RISK = os.getenv("DEFAULT_RISK", "moderate").strip().lower()
PUSH_HOUR_UTC = int(os.getenv("PUSH_HOUR_UTC", "15")
                    )          # daily snapshot push time
# alerts check interval (minutes)
ALERT_MINUTES = int(os.getenv("ALERT_EVERY_MIN", "15"))
CMC_API_KEY = os.getenv("CMC_API_KEY", "").strip()

# OKX instruments
OKX_FUTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP",
            "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]
OKX_SPOT_ETHBTC = "ETH-BTC"

# Alt basket (equal-weight momentum “index” on weekly candles)
ALT_BASKET = ["ADA-USDT", "BNB-USDT", "AVAX-USDT", "LINK-USDT",
              "MATIC-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT"]

# Subscribers kept in-memory (add your chat ID with /start)
SUBSCRIBERS: set[int] = set()

# Risk profiles tweak thresholds (lower = more cautious)
RISK_MULT = {
    "conservative": 0.9,
    "moderate": 1.0,
    "aggressive": 1.15
}

# --------------------- Utility & TA ------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)


def stoch_rsi(rsi_series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    min_rsi = rsi_series.rolling(period).min()
    max_rsi = rsi_series.rolling(period).max()
    stoch = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-9)
    k = stoch.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.fillna(0), d.fillna(0)


def two_week_downsample(weekly_close: pd.Series) -> pd.Series:
    idx = np.arange(len(weekly_close)) // 2
    close_2w = weekly_close.groupby(idx).last()
    close_2w.index = range(len(close_2w))
    return close_2w


def fib_extension_proximity(close_series: pd.Series, lookback_weeks: int = 52) -> Tuple[float, float]:
    s = close_series.dropna()
    if len(s) < lookback_weeks + 5:
        return (float("nan"), float("nan"))
    s_win = s[-lookback_weeks:]
    lo = float(s_win.min())
    hi = float(s_win.max())
    if hi <= 0 or hi == lo:
        return (float("nan"), float("nan"))
    ext_1272 = hi + 0.272 * (hi - lo)
    last = float(s.iloc[-1])
    pct_away = (ext_1272 - last) / ext_1272 * 100.0
    return ext_1272, pct_away


def tri_flag(value: Optional[float], warn: float, flag: float, higher_is_risk: bool = True) -> Tuple[str, str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "⚪", "na"
    v = float(value)
    if higher_is_risk:
        if v >= flag:
            return "🔴", "red"
        if v >= warn:
            return "🟡", "yellow"
        return "🟢", "green"
    else:
        if v <= flag:
            return "🔴", "red"
        if v <= warn:
            return "🟡", "yellow"
        return "🟢", "green"


def color_dot(band: str) -> str:
    return {"red": "🔴", "yellow": "🟡", "green": "🟢"}.get(band, "⚪")


def fmt_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.2f}T"
    if absx >= 1e9:
        return f"${x/1e9:.2f}B"
    if absx >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"


def safe_float(x, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

# ---------------------- HTTP Data Client -------------------


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.client.aclose()

    async def cg_global(self) -> Dict[str, Any]:
        r = await self.client.get("https://api.coingecko.com/api/v3/global")
        r.raise_for_status()
        return r.json()["data"]

    async def cmc_global(self) -> Optional[Dict[str, Any]]:
        if not CMC_API_KEY:
            return None
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
        url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
        r = await self.client.get(url, headers=headers)
        if r.status_code != 200:
            return None
        return r.json()["data"]

    async def okx_get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        url = f"https://www.okx.com{path}"
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "0":
            raise RuntimeError(f"OKX error: {js.get('msg')}")
        return js

    async def okx_funding(self, instId: str) -> Optional[float]:
        try:
            js = await self.okx_get("/api/v5/public/funding-rate", {"instId": instId})
            data = js.get("data", [])
            if not data:
                return None
            return safe_float(data[0].get("fundingRate"))
        except Exception as e:
            log.warning("funding %s failed: %s", instId, e)
            return None

    async def okx_ticker(self, instId: str) -> Optional[Dict[str, Any]]:
        try:
            js = await self.okx_get("/api/v5/market/ticker", {"instId": instId})
            data = js.get("data", [])
            return data[0] if data else None
        except Exception as e:
            log.warning("ticker %s failed: %s", instId, e)
            return None

    async def okx_oi_usd(self, instId: str) -> Optional[float]:
        try:
            js = await self.okx_get("/api/v5/public/open-interest", {"instId": instId, "instType": "SWAP"})
            data = js.get("data", [])
            if not data:
                return None
            row = data[0]
            oi_ccy = safe_float(row.get("oiCcy"))
            last = await self.okx_ticker(instId)
            last_px = safe_float(last.get("last")) if last else None
            if oi_ccy is not None and last_px is not None:
                return oi_ccy * last_px
            oi = safe_float(row.get("oi"))
            ctVal = safe_float(row.get("ctVal"))
            if oi is not None and ctVal is not None and last_px is not None:
                return oi * ctVal * last_px
            return None
        except Exception as e:
            log.warning("open interest %s failed: %s", instId, e)
            return None

    async def okx_candles(self, instId: str, bar: str, limit: int) -> pd.DataFrame:
        js = await self.okx_get("/api/v5/market/candles", {"instId": instId, "bar": bar, "limit": str(limit)})
        raw = js.get("data", [])
        rows = []
        for r in raw:
            ts = int(r[0])
            o = float(r[1])
            h = float(r[2])
            l = float(r[3])
            c = float(r[4])
            rows.append((ts, o, h, l, c))
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"]).sort_values(
            "ts").reset_index(drop=True)
        return df

    async def google_trends_score(self, kw: List[str]) -> Optional[float]:
        def _work():
            tr = TrendReq(hl="en-US", tz=0)
            tr.build_payload(kw, timeframe="now 7-d")
            df = tr.interest_over_time()
            if df.empty:
                return None
            vals = df[kw].mean(axis=1)
            return float(vals.mean())
        try:
            return await asyncio.to_thread(_work)
        except Exception as e:
            log.warning("pytrends failed: %s", e)
            return None

    async def fear_greed(self, limit: int = 1) -> List[Dict[str, Any]]:
        r = await self.client.get(f"https://api.alternative.me/fng/?limit={limit}")
        r.raise_for_status()
        return r.json().get("data", [])

# --------------- Metrics Gathering & Computation -----------


async def gather_metrics(dc: DataClient, risk: str) -> Dict[str, Any]:
    cmc_task = asyncio.create_task(
        dc.cmc_global() if CMC_API_KEY else asyncio.sleep(0, result=None))
    cg_task = asyncio.create_task(dc.cg_global())

    ticker_ethbtc_t = asyncio.create_task(dc.okx_ticker(OKX_SPOT_ETHBTC))

    fund_tasks = {inst: asyncio.create_task(
        dc.okx_funding(inst)) for inst in OKX_FUTS}
    oi_btc_t = asyncio.create_task(dc.okx_oi_usd("BTC-USDT-SWAP"))
    oi_eth_t = asyncio.create_task(dc.okx_oi_usd("ETH-USDT-SWAP"))

    trends_t = asyncio.create_task(
        dc.google_trends_score(["crypto", "bitcoin", "ethereum"]))

    fng_now_t = asyncio.create_task(dc.fear_greed(limit=1))
    fng_hist_t = asyncio.create_task(dc.fear_greed(limit=60))

    btc_day_t = asyncio.create_task(dc.okx_candles("BTC-USDT", "1D", 500))
    btc_w_t = asyncio.create_task(dc.okx_candles("BTC-USDT", "1W", 400))
    ethbtc_w_t = asyncio.create_task(dc.okx_candles("ETH-BTC", "1W", 400))
    alt_week_tasks = {sym: asyncio.create_task(
        dc.okx_candles(sym, "1W", 400)) for sym in ALT_BASKET}

    cmc = await cmc_task
    cg = await cg_task

    if cmc:
        totalcap = safe_float(cmc.get("quote", {}).get(
            "USD", {}).get("total_market_cap"))
        btc_dom = safe_float(cmc.get("btc_dominance"))
    else:
        totalcap = safe_float(cg.get("total_market_cap", {}).get("usd"))
        btc_dom = safe_float(cg.get("market_cap_percentage", {}).get("btc"))

    ethbtc_tick = await ticker_ethbtc_t
    eth_btc = safe_float(ethbtc_tick.get("last")) if ethbtc_tick else None

    altcap_ratio = None
    if totalcap is not None and btc_dom is not None:
        altcap = totalcap * (1.0 - btc_dom/100.0)
        btc_mcap = totalcap * (btc_dom/100.0)
        if btc_mcap > 0:
            altcap_ratio = altcap / btc_mcap

    funding = {inst: await ft for inst, ft in fund_tasks.items()}
    fund_vals = [v for v in funding.values() if v is not None]
    fund_abs = [abs(100*v) for v in fund_vals]
    fund_max = max(fund_abs) if fund_abs else None
    fund_med = float(np.median(fund_abs)) if fund_abs else None
    top3 = []
    if fund_vals:
        pairs = list(funding.items())
        pairs_pct = [(k, abs(100.0*v) if v is not None else None)
                     for k, v in pairs]
        pairs_pct = [p for p in pairs_pct if p[1] is not None]
        pairs_pct.sort(key=lambda x: x[1], reverse=True)
        top3 = pairs_pct[:3]

    oi_btc = await oi_btc_t
    oi_eth = await oi_eth_t

    trends_score = await trends_t
    fng_now = await fng_now_t
    fng_hist = await fng_hist_t

    fng_val = None
    fng_ma14 = None
    fng_ma30 = None
    greed_streak = 0
    greed_pct30 = None
    if fng_now:
        fng_val = safe_float(fng_now[0].get("value"))
    if fng_hist:
        vals = [safe_float(x.get("value"))
                for x in fng_hist if safe_float(x.get("value")) is not None]
        if vals:
            ser = pd.Series(list(reversed(vals)))
            fng_ma14 = float(ser.rolling(14).mean(
            ).iloc[-1]) if len(ser) >= 14 else None
            fng_ma30 = float(ser.rolling(30).mean(
            ).iloc[-1]) if len(ser) >= 30 else None
            streak = 0
            for v in reversed(vals):
                if v is not None and v >= 70:
                    streak += 1
                else:
                    break
            greed_streak = streak
            last30 = [v for v in vals[:30] if v is not None]
            if last30:
                greed_pct30 = 100.0 * \
                    sum(1 for v in last30 if v >= 70) / len(last30)

    btc_day = await btc_day_t
    btc_week = await btc_w_t
    ethbtc_week = await ethbtc_w_t
    alt_weeks = {sym: await alt_week_tasks[sym] for sym in ALT_BASKET}

    # Pi Cycle (OKX daily): 111SMA vs 2*350SMA
    pi_prox = None
    pi_crossed = False
    days_since_cross = None
    try:
        closes = btc_day["close"]
        sma111 = closes.rolling(111).mean()
        sma350 = closes.rolling(350).mean()
        line = sma350 * 2.0
        if len(closes) >= 350:
            last_ratio = float(
                sma111.iloc[-1] / (line.iloc[-1] + 1e-9)) * 100.0
            pi_prox = last_ratio
            above = (sma111 > line)
            if above.iloc[-1]:
                pi_crossed = True
                cross_idx = None
                for i in range(len(above)-2, -1, -1):
                    if not above.iloc[i]:
                        cross_idx = i + 1
                        break
                if cross_idx is None:
                    cross_idx = 0
                days_since_cross = int((len(above)-1) - cross_idx)
    except Exception as e:
        log.warning("Pi cycle computation failed: %s", e)

    # Weekly & 2W momentum
    btc_week_close = btc_week["close"]
    ethbtc_week_close = ethbtc_week["close"]

    btc_2w = two_week_downsample(btc_week_close)
    ethbtc_2w = two_week_downsample(ethbtc_week_close)

    btc_rsi_2w = rsi(btc_2w, 14)
    ethbtc_rsi_2w = rsi(ethbtc_2w, 14)

    btc_rsi_ma = btc_rsi_2w.rolling(10).mean()
    ethbtc_rsi_ma = ethbtc_rsi_2w.rolling(10).mean()

    btc_k_2w, btc_d_2w = stoch_rsi(btc_rsi_2w, 14, 3, 3)

    # ALT equal-weight weekly index
    alt_df = pd.DataFrame()
    for sym, df in alt_weeks.items():
        alt_df[sym] = df["close"].values
    if not alt_df.empty:
        min_len = min(len(alt_df[sym]) for sym in alt_df.columns)
        for sym in alt_df.columns:
            alt_df[sym] = alt_df[sym].iloc[-min_len:].reset_index(drop=True)
        norm = alt_df / alt_df.iloc[0]
        alt_index = norm.mean(axis=1)
    else:
        alt_index = pd.Series(dtype=float)

    if len(alt_index) > 0:
        alt_index_2w = two_week_downsample(pd.Series(alt_index))
        alt_rsi_2w = rsi(alt_index_2w, 14)
        alt_rsi_ma = alt_rsi_2w.rolling(10).mean()
        alt_k_2w, alt_d_2w = stoch_rsi(alt_rsi_2w, 14, 3, 3)
    else:
        alt_rsi_2w = pd.Series(dtype=float)
        alt_rsi_ma = pd.Series(dtype=float)
        alt_k_2w = pd.Series(dtype=float)
        alt_d_2w = pd.Series(dtype=float)

    btc_ext, btc_pct_away = fib_extension_proximity(
        btc_week_close, lookback_weeks=52)
    alt_ext, alt_pct_away = fib_extension_proximity(pd.Series(
        alt_index), lookback_weeks=52) if len(alt_index) else (float("nan"), float("nan"))

    out: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "risk": risk,

        "totalcap": totalcap,
        "btc_dom": btc_dom,
        "eth_btc": eth_btc,
        "altcap_btc_ratio": altcap_ratio,

        "funding": funding,
        "funding_max_pct": fund_max,
        "funding_med_pct": fund_med,
        "funding_top3": top3,

        "oi_btc_usd": oi_btc,
        "oi_eth_usd": oi_eth,

        "trends_avg7": trends_score,

        "fng": {
            "value": fng_val,
            "ma14": fng_ma14,
            "ma30": fng_ma30,
            "streak_greed_days": greed_streak,
            "pct30_greed": greed_pct30
        },

        "pi": {
            "prox_pct": pi_prox,           # 100% == cross
            "crossed": pi_crossed,
            "days_since_cross": days_since_cross
        },

        "ta": {
            "btc": {
                "rsi_2w": float(btc_rsi_2w.iloc[-1]) if len(btc_rsi_2w) else None,
                "rsi_2w_ma": float(btc_rsi_ma.iloc[-1]) if len(btc_rsi_ma) else None,
                "k_2w": float(btc_k_2w.iloc[-1]) if len(btc_k_2w) else None,
                "d_2w": float(btc_d_2w.iloc[-1]) if len(btc_d_2w) else None,
                "fib1272": btc_ext,
                "fib_pct_away": btc_pct_away
            },
            "ethbtc": {
                "rsi_2w": float(ethbtc_rsi_2w.iloc[-1]) if len(ethbtc_rsi_2w) else None,
                "rsi_2w_ma": float(ethbtc_rsi_ma.iloc[-1]) if len(ethbtc_rsi_ma) else None,
            },
            "alt": {
                "rsi_2w": float(alt_rsi_2w.iloc[-1]) if len(alt_rsi_2w) else None,
                "rsi_2w_ma": float(alt_rsi_ma.iloc[-1]) if len(alt_rsi_ma) else None,
                "k_2w": float(alt_k_2w.iloc[-1]) if len(alt_k_2w) else None,
                "d_2w": float(alt_d_2w.iloc[-1]) if len(alt_d_2w) else None,
                "fib1272": alt_ext,
                "fib_pct_away": alt_pct_away
            }
        }
    }
    return out

# --------------- Scoring & Formatting ----------------------


def score_from_band(band: str, green: int, yellow: int, red: int) -> int:
    if band == "green":
        return green
    if band == "yellow":
        return yellow
    if band == "red":
        return red
    return 0


def composite_score(m: Dict[str, Any]) -> Tuple[int, List[Tuple[str, int]]]:
    t = m.get("ta", {})
    fng = m.get("fng", {})
    risk = m.get("risk", "moderate")
    mult = RISK_MULT.get(risk, 1.0)

    drivers: List[Tuple[str, int]] = []

    def adj(x): return x * mult

    # Market structure
    band_btc_dom = tri_flag(m.get("btc_dom"), warn=48.0,
                            flag=60.0, higher_is_risk=True)[1]
    drivers.append(("BTC dominance", score_from_band(band_btc_dom, 0, 50, 90)))

    band_ethbtc = tri_flag(m.get("eth_btc"), warn=0.072,
                           flag=0.09, higher_is_risk=True)[1]
    drivers.append(("ETH/BTC", score_from_band(band_ethbtc, 0, 40, 80)))

    band_altr = tri_flag(m.get("altcap_btc_ratio"), warn=adj(
        1.44), flag=adj(1.80), higher_is_risk=True)[1]
    drivers.append(("Alt/BTC mcap", score_from_band(band_altr, 0, 60, 90)))

    # Derivatives
    band_fund = tri_flag(m.get("funding_max_pct"), warn=adj(
        0.08), flag=adj(0.10), higher_is_risk=True)[1]
    drivers.append(("Funding (max)", score_from_band(band_fund, 0, 25, 55)))

    band_oi_btc = tri_flag(m.get("oi_btc_usd"), warn=16e9,
                           flag=20e9, higher_is_risk=True)[1]
    drivers.append(("BTC OI", score_from_band(band_oi_btc, 0, 35, 65)))

    band_oi_eth = tri_flag(m.get("oi_eth_usd"), warn=6.4e9,
                           flag=8e9, higher_is_risk=True)[1]
    drivers.append(("ETH OI", score_from_band(band_oi_eth, 0, 25, 55)))

    # Sentiment
    band_trends = tri_flag(m.get("trends_avg7"), warn=60.0,
                           flag=75.0, higher_is_risk=True)[1]
    drivers.append(("Google Trends", score_from_band(band_trends, 0, 30, 60)))

    band_fng = tri_flag(fng.get("value"), warn=56.0,
                        flag=70.0, higher_is_risk=True)[1]
    drivers.append(("F&G (today)", score_from_band(band_fng, 0, 50, 85)))

    band_fng14 = tri_flag(fng.get("ma14"), warn=56.0,
                          flag=70.0, higher_is_risk=True)[1]
    drivers.append(("F&G 14d", score_from_band(band_fng14, 0, 50, 85)))

    band_fng30 = tri_flag(fng.get("ma30"), warn=52.0,
                          flag=65.0, higher_is_risk=True)[1]
    drivers.append(("F&G 30d", score_from_band(band_fng30, 0, 60, 90)))

    streak = fng.get("streak_greed_days") or 0
    pct30 = fng.get("pct30_greed")
    band_streak = tri_flag(streak, warn=8, flag=10, higher_is_risk=True)[1]
    band_persist = tri_flag(pct30, warn=48.0, flag=60.0,
                            higher_is_risk=True)[1]
    persist_score = max(score_from_band(band_streak, 0, 50, 90),
                        score_from_band(band_persist, 0, 50, 90))
    drivers.append(("F&G persistence", persist_score))

    # Pi cycle
    pi = m.get("pi", {})
    band_pi = tri_flag(pi.get("prox_pct"), warn=90.0,
                       flag=100.0, higher_is_risk=True)[1]
    drivers.append(("Pi Cycle", score_from_band(band_pi, 0, 60, 95)))

    # 2W RSI & Stoch
    btc_rsi = t["btc"].get("rsi_2w")
    btc_rsi_ma = t["btc"].get("rsi_2w_ma")
    band_btc_rsi = tri_flag(
        btc_rsi, warn=60.0, flag=70.0, higher_is_risk=True)[1]
    rsi_loss_ma = (
        btc_rsi is not None and btc_rsi_ma is not None and btc_rsi < btc_rsi_ma)
    drivers.append(("BTC RSI 2W", min(100, score_from_band(
        band_btc_rsi, 0, 35, 70) + (10 if rsi_loss_ma else 0))))

    ethbtc_rsi = t["ethbtc"].get("rsi_2w")
    ethbtc_rsi_ma = t["ethbtc"].get("rsi_2w_ma")
    band_ethbtc_rsi = tri_flag(
        ethbtc_rsi, warn=55.0, flag=65.0, higher_is_risk=True)[1]
    eth_loss_ma = (
        ethbtc_rsi is not None and ethbtc_rsi_ma is not None and ethbtc_rsi < ethbtc_rsi_ma)
    drivers.append(("ETH/BTC RSI 2W", min(100, score_from_band(band_ethbtc_rsi,
                   0, 35, 70) + (10 if eth_loss_ma else 0))))

    alt_rsi = t["alt"].get("rsi_2w")
    alt_rsi_ma = t["alt"].get("rsi_2w_ma")
    band_alt_rsi = tri_flag(
        alt_rsi, warn=65.0, flag=75.0, higher_is_risk=True)[1]
    alt_loss_ma = (
        alt_rsi is not None and alt_rsi_ma is not None and alt_rsi < alt_rsi_ma)
    drivers.append(("ALT RSI 2W", min(100, score_from_band(
        band_alt_rsi, 0, 30, 60) + (10 if alt_loss_ma else 0))))

    # Fibonacci proximity
    band_btc_fib = tri_flag(t["btc"].get(
        "fib_pct_away"), warn=3.0, flag=1.5, higher_is_risk=False)[1]
    drivers.append(("BTC Fib 1.272", score_from_band(band_btc_fib, 0, 35, 70)))

    band_alt_fib = tri_flag(t["alt"].get(
        "fib_pct_away"), warn=3.0, flag=1.5, higher_is_risk=False)[1]
    drivers.append(("ALT Fib 1.272", score_from_band(band_alt_fib, 0, 35, 70)))

    drivers_sorted = sorted(drivers, key=lambda x: x[1], reverse=True)
    take = drivers_sorted[:10]
    total = int(round(sum(s for _, s in take) / len(take))) if take else 0
    return min(100, max(0, total)), drivers_sorted


def top5_lines(drivers_sorted: List[Tuple[str, int]]) -> List[str]:
    lines = []
    for name, sc in drivers_sorted[:5]:
        band = "green" if sc < 40 else ("yellow" if sc < 70 else "red")
        lines.append(f"• {name}: {color_dot(band)} {sc}/100")
    return lines


def format_snapshot(m: Dict[str, Any]) -> str:
    risk = m.get("risk", "moderate")
    t = m.get("ta", {})
    fng = m.get("fng", {})

    btc_dom = m.get("btc_dom")
    eth_btc = m.get("eth_btc")
    alt_ratio = m.get("altcap_btc_ratio")

    b1, bb1 = tri_flag(btc_dom, 48.0, 60.0, True)
    b2, bb2 = tri_flag(eth_btc, 0.072, 0.09, True)
    b3, bb3 = tri_flag(alt_ratio, 1.44, 1.80, True)

    fund_max = m.get("funding_max_pct")
    fund_med = m.get("funding_med_pct")
    top3 = m.get("funding_top3", [])
    b4, bb4 = tri_flag(fund_max, 0.08, 0.10, True)

    oi_btc = m.get("oi_btc_usd")
    oi_eth = m.get("oi_eth_usd")
    b5, bb5 = tri_flag(oi_btc, 16e9, 20e9, True)
    b6, bb6 = tri_flag(oi_eth, 6.4e9, 8e9, True)

    trends = m.get("trends_avg7")
    b7, bb7 = tri_flag(trends, 60.0, 75.0, True)

    fng_val = fng.get("value")
    fng_ma14 = fng.get("ma14")
    fng_ma30 = fng.get("ma30")
    streak = fng.get("streak_greed_days")
    pct30 = fng.get("pct30_greed")

    b8, _ = tri_flag(fng_val, 56.0, 70.0, True)
    b9, _ = tri_flag(fng_ma14, 56.0, 70.0, True)
    b10, _ = tri_flag(fng_ma30, 52.0, 65.0, True)

    pi = m.get("pi", {})
    pi_prox = pi.get("prox_pct")
    pi_crossed = pi.get("crossed")
    days_cross = pi.get("days_since_cross")
    bpi, _ = tri_flag(pi_prox, 90.0, 100.0, True)

    btc = t.get("btc", {})
    ethbtc = t.get("ethbtc", {})
    alt = t.get("alt", {})

    btc_rsi = btc.get("rsi_2w")
    btc_rsi_ma = btc.get("rsi_2w_ma")
    eth_rsi = ethbtc.get("rsi_2w")
    eth_rsi_ma = ethbtc.get("rsi_2w_ma")
    alt_rsi = alt.get("rsi_2w")
    alt_rsi_ma = alt.get("rsi_2w_ma")

    btc_k = btc.get("k_2w")
    btc_d = btc.get("d_2w")
    alt_k = alt.get("k_2w")
    alt_d = alt.get("d_2w")

    btc_fib = btc.get("fib1272")
    btc_fib_away = btc.get("fib_pct_away")
    alt_fib = alt.get("fib1272")
    alt_fib_away = alt.get("fib_pct_away")

    total, drivers_sorted = composite_score(m)
    band_total = "green" if total < 40 else ("yellow" if total < 70 else "red")
    comp_flag = color_dot(band_total)

    lines = []
    ts = m.get("ts", "")
    lines.append(f"📊 Crypto Market Snapshot — {ts} UTC")
    lines.append(f"Profile: {risk}")
    lines.append("")
    lines.append("Market Structure")
    lines.append(
        f"• Bitcoin market share of total crypto: {b1} {btc_dom:.2f}%  (warn ≥ 48.00%, flag ≥ 60.00%)" if btc_dom is not None else "• Bitcoin market share of total crypto: n/a")
    lines.append(
        f"• Ether price relative to Bitcoin (ETH/BTC): {b2} {eth_btc:.5f}  (warn ≥ 0.07200, flag ≥ 0.09000)" if eth_btc is not None else "• Ether price relative to Bitcoin (ETH/BTC): n/a")
    lines.append(
        f"• Altcoin market cap / Bitcoin market cap: {b3} {alt_ratio:.2f}  (warn ≥ 1.44, flag ≥ 1.80)" if alt_ratio is not None else "• Altcoin market cap / Bitcoin market cap: n/a")
    lines.append("")
    lines.append("Derivatives")
    if fund_max is not None and fund_med is not None:
        lines.append(
            f"• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — max: {b4} {fund_max:.3f}% | median: {color_dot(bb4)} {fund_med:.3f}%  (warn ≥ 0.080%, flag ≥ 0.100%)")
    else:
        lines.append("• Funding (basket): n/a")
    if top3:
        tops = ", ".join(
            [f"{k.replace('-USDT-SWAP','USDT')} {v:.3f}%" for k, v in top3])
        lines.append(f"  Top-3 funding extremes: {tops}")
    lines.append(
        f"• Bitcoin open interest (USD): {b5} {fmt_money(oi_btc)}  (warn ≥ $16,000,000,000, flag ≥ $20,000,000,000)")
    lines.append(
        f"• Ether open interest (USD): {b6} {fmt_money(oi_eth)}  (warn ≥ $6,400,000,000, flag ≥ $8,000,000,000)")
    lines.append("")
    lines.append("Sentiment")
    lines.append(
        f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {b7} {trends:.1f}  (warn ≥ 60.0, flag ≥ 75.0)" if trends is not None else "• Google Trends avg: n/a")
    if fng_val is not None:
        lines.append(
            f"• Fear & Greed Index (overall crypto): {b8} {int(fng_val)}  (warn ≥ 56, flag ≥ 70)")
    if fng_ma14 is not None:
        lines.append(
            f"• Fear & Greed 14-day average: {b9} {fng_ma14:.1f}  (warn ≥ 56, flag ≥ 70)")
    if fng_ma30 is not None:
        lines.append(
            f"• Fear & Greed 30-day average: {b10} {fng_ma30:.1f}  (warn ≥ 52, flag ≥ 65)")
    if (streak is not None) and (pct30 is not None):
        lines.append(f"• Greed persistence: {('🔴' if streak>=10 or (pct30 is not None and pct30>=60) else ('🟡' if streak>=8 or (pct30 is not None and pct30>=48) else '🟢'))} {streak} days in a row | {pct30:.0f}% of last 30 days ≥ 70  (warn: days ≥ 8 or pct ≥ 48%; flag: days ≥ 10 or pct ≥ 60%)")
    lines.append("")
    lines.append("Cycle & On-Chain")
    if pi_prox is not None:
        suffix = " (crossed)" if pi_crossed else " of trigger (100% = cross)"
        lines.append(f"• Pi Cycle Top proximity: {bpi} {pi_prox:.1f}%{suffix}")
        if pi_crossed and days_cross is not None:
            lines.append(f"  Days since cross: {days_cross}")
    else:
        lines.append("• Pi Cycle Top proximity: ⚪ n/a")
    lines.append("")
    lines.append("Momentum (2W) & Extensions (1W)")
    if (btc_rsi is not None) and (btc_rsi_ma is not None):
        lines.append(
            f"• BTC RSI (2W): {('🔴' if btc_rsi>=70 else '🟡' if btc_rsi>=60 else '🟢')} {btc_rsi:.1f} (MA {btc_rsi_ma:.1f}) (warn ≥ 60.0, flag ≥ 70.0)")
    if (eth_rsi is not None) and (eth_rsi_ma is not None):
        lines.append(
            f"• ETH/BTC RSI (2W): {('🔴' if eth_rsi>=65 else '🟡' if eth_rsi>=55 else '🟢')} {eth_rsi:.1f} (MA {eth_rsi_ma:.1f}) (warn ≥ 55.0, flag ≥ 65.0)")
    if (btc_k is not None) and (btc_d is not None):
        lines.append(
            f"• BTC Stoch RSI (2W) K/D: {('🔴' if btc_k<btc_d and max(btc_k,btc_d)>=0.80 else '🟢')} {btc_k:.2f}/{btc_d:.2f} (overbought ≥ 0.80; red = bearish cross from OB)")
    if (btc_fib is not None) and (not math.isnan(btc_fib)):
        if btc_fib_away is not None:
            lines.append(
                f"• BTC Fibonacci extension proximity: {('🔴' if btc_fib_away<=1.5 else '🟡' if btc_fib_away<=3.0 else '🟢')} 1.272 @ {btc_fib_away:.2f}% away (warn ≤ 3.0%, flag ≤ 1.5%)")
        else:
            lines.append("• BTC Fibonacci extension proximity: n/a")
    if (alt_rsi is not None) and (alt_rsi_ma is not None):
        lines.append(
            f"• ALT basket (equal-weight) RSI (2W): {('🔴' if alt_rsi>=75 else '🟡' if alt_rsi>=65 else '🟢')} {alt_rsi:.1f} (MA {alt_rsi_ma:.1f}) (warn ≥ 65.0, flag ≥ 75.0)")
    if (alt_k is not None) and (alt_d is not None):
        lines.append(
            f"• ALT basket Stoch RSI (2W) K/D: {('🔴' if alt_k<alt_d and max(alt_k,alt_d)>=0.80 else '🟢')} {alt_k:.2f}/{alt_d:.2f} (overbought ≥ 0.80; red = bearish cross from OB)")
    if (alt_fib is not None) and (not math.isnan(alt_fib)):
        if alt_fib_away is not None:
            lines.append(
                f"• ALT basket Fibonacci proximity: {('🔴' if alt_fib_away<=1.5 else '🟡' if alt_fib_away<=3.0 else '🟢')} 1.272 @ {alt_fib_away:.2f}% away (warn ≤ 3.0%, flag ≤ 1.5%)")
        else:
            lines.append("• ALT basket Fibonacci proximity: n/a")

    lines.append("")
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {comp_flag} {total}/100 (yellow ≥ 40, red ≥ 70)")
    lines.append("• Top drivers:")
    lines.extend(top5_lines(drivers_sorted))

    triggered: List[str] = []
    if bb1 == "red":
        triggered.append("BTC dominance elevated")
    if bb2 == "red":
        triggered.append("ETH/BTC elevated")
    if bb3 == "red":
        triggered.append("Alt market cap relative to BTC high")
    if bb4 == "red":
        triggered.append("Funding extreme")
    if bb5 == "red":
        triggered.append("BTC OI high")
    if bb6 == "red":
        triggered.append("ETH OI high")
    if bb7 == "red":
        triggered.append("Google Trends elevated")
    if (fng_ma30 is not None) and (fng_ma30 >= 65):
        triggered.append("F&G 30-day avg in Greed")
    if ((streak is not None) and streak >= 10) or ((pct30 is not None) and pct30 >= 60):
        triggered.append("Greed persistence over threshold")
    if (pi_prox is not None) and (pi_prox >= 100.0):
        triggered.append("Pi Cycle cross")

    if triggered:
        lines.append("")
        lines.append(
            f"⚠️ Triggered flags ({len(triggered)}): " + ", ".join(triggered))

    return "\n".join(lines)


# ----------------------- Bot Handlers ----------------------
USER_RISK: Dict[int, str] = {}


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    SUBSCRIBERS.add(update.effective_chat.id)
    if update.effective_user:
        USER_RISK[update.effective_user.id] = USER_RISK.get(
            update.effective_user.id, DEFAULT_RISK)
    msg = (
        "Hi! I’ll track market-cycle metrics for you.\n\n"
        "Commands:\n"
        "• /status — current snapshot\n"
        "• /setrisk <conservative|moderate|aggressive> — tune thresholds\n\n"
        f"Daily summary at {PUSH_HOUR_UTC:02d}:00 UTC and alerts every {ALERT_MINUTES} min."
    )
    await update.message.reply_text(msg)


async def cmd_setrisk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /setrisk <conservative|moderate|aggressive>")
        return
    val = context.args[0].strip().lower()
    if val not in RISK_MULT:
        await update.message.reply_text("Pick one of: conservative, moderate, aggressive")
        return
    USER_RISK[update.effective_user.id] = val
    await update.message.reply_text(f"Okay — risk profile set to '{val}'.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    risk = USER_RISK.get(update.effective_user.id,
                         DEFAULT_RISK) if update.effective_user else DEFAULT_RISK
    async with DataClient() as dc:
        try:
            m = await gather_metrics(dc, risk)
            text = format_snapshot(m)
            await update.message.reply_text(text)
        except Exception as e:
            log.exception("status failed")
            await update.message.reply_text(f"⚠️ Could not fetch metrics right now: {e}")

# -------------------- Pushers / Schedulers -----------------


async def push_summary(app: Application):
    async with DataClient() as dc:
        for chat_id in list(SUBSCRIBERS):
            try:
                risk = DEFAULT_RISK
                m = await gather_metrics(dc, risk)
                txt = format_snapshot(m)
                await app.bot.send_message(chat_id=chat_id, text=txt)
            except Exception as e:
                log.warning("push_summary to %s failed: %s", chat_id, e)


async def push_alerts(app: Application):
    async with DataClient() as dc:
        try:
            m = await gather_metrics(dc, DEFAULT_RISK)
            txt = format_snapshot(m)
            if "⚠️ Triggered flags (" in txt:
                for chat_id in list(SUBSCRIBERS):
                    await app.bot.send_message(chat_id=chat_id, text="⏰ Alert check\n\n" + txt)
        except Exception as e:
            log.warning("push_alerts failed: %s", e)

# --------------------------- Health ------------------------


async def health(request):
    return web.json_response({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


async def run_health_only():
    app_web = web.Application()
    app_web.router.add_get("/health", health)
    runner = web.AppRunner(app_web)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=8080)
    await site.start()
    log.error("TELEGRAM_BOT_TOKEN not set. Running in health-only mode on :8080.")
    # Keep process alive
    await asyncio.Event().wait()

# --------------------------- Main --------------------------


async def main():
    if not BOT_TOKEN:
        await run_health_only()
        return

    # Telegram app
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("setrisk", cmd_setrisk))

    # Health server (Koyeb probes)
    app_web = web.Application()
    app_web.router.add_get("/health", health)
    runner = web.AppRunner(app_web)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=8080)
    await site.start()
    log.info("Health server listening on :8080")

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=str(get_localzone()))
    scheduler.add_job(lambda: asyncio.create_task(push_summary(app)),
                      "cron", hour=PUSH_HOUR_UTC, minute=0, second=0, id="push_summary", coalesce=True, max_instances=1)
    scheduler.add_job(lambda: asyncio.create_task(push_alerts(app)),
                      "cron", minute=f"*/{ALERT_MINUTES}", id="push_alerts", coalesce=True, max_instances=1)
    scheduler.start()

    log.info("Bot running. Press Ctrl+C to exit.")
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
