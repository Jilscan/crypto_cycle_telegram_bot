# main.py
# Crypto Cycle Telegram Bot — single-file version with Top-5 contributors
# Features:
# - Risk profiles: conservative / moderate / aggressive
# - Daily summary + 15-min alerts (UTC)
# - Market structure (BTC dominance, ETH/BTC, Alt/BTC cap ratio)
# - Derivatives (funding basket, BTC & ETH OI USD from OKX)
# - Sentiment (Google Trends 7d avg; Fear & Greed today/14d/30d; Greed persistence)
# - Cycle (Pi Cycle ratio from OKX daily candles, not CoinGecko)
# - Momentum (2W RSI + “losing RSI MA”; 2W StochRSI K/D)
# - Fibonacci proximity (1.272 on BTC 1W and equal-weight alt basket 1W)
# - Composite Alt-Top Certainty 0–100 with Top-5 drivers
#
# Environment:
#   TELEGRAM_TOKEN=<token>                (required)
#   DEFAULT_PROFILE=conservative|moderate|aggressive (optional, default moderate)
#   ALLOWED_CHAT_IDS=<id1,id2,...>       (optional; if set, restricts chat access)
#   CMC_API_KEY=<key>                    (optional; if set, uses CMC for dominance/alt ratio)
#
# Notes:
# - Uses HTML parse_mode safely (no angle brackets in text other than <b>/<i>).
# - Health endpoint on :8080 for Koyeb.
# - No CoinGecko daily series used for Pi Cycle; OKX replaces it.

from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update
import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from aiohttp import web

# Google Trends (pytrends)
from pytrends.request import TrendReq

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("crypto-cycle-bot")

# ---------------- Config ----------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "").strip()
if not TELEGRAM_TOKEN:
    raise SystemExit("TELEGRAM_TOKEN is required")

DEFAULT_PROFILE = os.environ.get("DEFAULT_PROFILE", "moderate").lower().strip()
ALLOWED_CHAT_IDS = {i for i in os.environ.get(
    "ALLOWED_CHAT_IDS", "").split(",") if i.strip()}
CMC_API_KEY = os.environ.get("CMC_API_KEY", "").strip()

# Symbols
ALT_BASKET = ["SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
              "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"]

# Schedules (UTC)
DAILY_SUMMARY_CRON = "0 14 * * *"     # 14:00 UTC daily
ALERTS_CRON = "*/15 * * * *"          # every 15 minutes

# Emoji badges
GREEN = "🟢 "
YELLOW = "🟡 "
RED = "🔴 "

# ---------------- Risk Thresholds ----------------
BASE_THRESHOLDS = {
    # Market structure
    # ≥ 60% green->red boundary for tops (used as a flag condition)
    "btc_dominance_flag": 0.60,
    "btc_dominance_warn": 0.48,     # ≥ 48% warn
    "eth_btc_ratio_flag": 0.090,    # ETH/BTC ≥ 0.09 is toppy
    "eth_btc_ratio_warn": 0.072,
    "altcap_btc_ratio_flag": 1.80,  # (total - BTC) / BTC
    "altcap_btc_ratio_warn": 1.44,

    # Derivatives
    "funding_rate_abs_flag": 0.0010,  # 0.10%
    "funding_rate_abs_warn": 0.0008,  # 0.08%
    "oi_btc_usd_flag": 20_000_000_000,
    "oi_btc_usd_warn": 16_000_000_000,
    "oi_eth_usd_flag": 8_000_000_000,
    "oi_eth_usd_warn": 6_400_000_000,

    # Sentiment
    "google_trends_7d_flag": 75.0,
    "google_trends_7d_warn": 60.0,
    "fear_greed_flag": 70,
    "fear_greed_warn": 56,
    "fear_greed_ma14_flag": 70,
    "fear_greed_ma14_warn": 56,
    "fear_greed_ma30_flag": 65,
    "fear_greed_ma30_warn": 52,
    "greed_days_row_flag": 10,      # consecutive days ≥70
    "greed_days_row_warn": 8,
    "greed_pct30_flag": 0.60,       # share of last 30 days in greed ≥ 60%
    "greed_pct30_warn": 0.48,

    # Cycle
    "pi_cycle_ratio_flag": 1.00,    # 111DMA >= 2*350DMA
    "pi_cycle_ratio_warn": 0.98,

    # Momentum (2W RSI)
    "btc_2w_rsi_flag": 70.0,
    "btc_2w_rsi_warn": 60.0,
    "ethbtc_2w_rsi_flag": 65.0,
    "ethbtc_2w_rsi_warn": 55.0,
    "alt_2w_rsi_flag": 75.0,
    "alt_2w_rsi_warn": 65.0,

    # Fibonacci (proximity percentage to 1.272 on 1W)
    "fib_prox_flag": 0.015,  # ≤ 1.5%
    "fib_prox_warn": 0.030,  # ≤ 3.0%
}

PROFILE_MULTIPLIER = {
    # Multiplicative adjustment to make it easier (conservative) or harder (aggressive) to trigger
    "conservative": 0.9,  # lowers warn/flag → triggers earlier
    "moderate": 1.0,
    "aggressive": 1.1,    # raises warn/flag → triggers later
}

# Per-chat risk profile in memory
CHAT_PROFILE: Dict[int, str] = {}

# ---------------- Small Utils ----------------


def pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"


def usd(x: Optional[float]) -> str:
    return "n/a" if x is None else f"${x:,.0f}"


def badge(val: Optional[float], warn: float, flag: float, higher_is_risk=True) -> str:
    if val is None:
        return GREEN
    if not higher_is_risk:
        # invert scale
        val = -val
        warn = -warn
        flag = -flag
    if val >= flag:
        return RED
    if val >= warn:
        return YELLOW
    return GREEN


def sma_list(arr: List[float], n: int) -> List[float]:
    if len(arr) < n:
        return []
    out = []
    s = sum(arr[:n])
    out.append(s/n)
    for i in range(n, len(arr)):
        s += arr[i] - arr[i-n]
        out.append(s/n)
    return out


def sma(arr: List[float], n: int) -> Optional[float]:
    if len(arr) < n:
        return None
    return sum(arr[-n:]) / n


def rsi(series: List[float], length: int = 14) -> List[float]:
    if len(series) < length + 1:
        return []
    gains = []
    losses = []
    for i in range(1, length + 1):
        d = series[i] - series[i-1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains)/length
    al = sum(losses)/length
    out = []
    for i in range(length+1, len(series)):
        d = series[i] - series[i-1]
        g = max(d, 0.0)
        l = max(-d, 0.0)
        ag = (ag*(length-1) + g) / length
        al = (al*(length-1) + l) / length
        if al == 0:
            out.append(100.0)
        else:
            rs = ag/al
            out.append(100.0 - (100.0/(1.0+rs)))
    return out


def stoch_rsi(series: List[float], rsi_len=14, k_len=3, d_len=3) -> Tuple[List[float], List[float]]:
    r = rsi(series, rsi_len)
    if len(r) < rsi_len:
        return [], []
    st = []
    for i in range(rsi_len, len(r)+1):
        win = r[i-rsi_len:i]
        lo = min(win)
        hi = max(win)
        st.append(0.0 if hi == lo else (r[i-1]-lo)/(hi-lo))

    def sma_local(arr, n):
        if len(arr) < n:
            return []
        return [sum(arr[i-n:i])/n for i in range(n, len(arr)+1)]
    k = sma_local(st, k_len)
    d = sma_local(k, d_len)
    return k, d


def add_scaled(sub: Dict[str, int], key: str, value: Optional[float], warn: float, flag: float, higher_is_risk=True) -> Tuple[float, float]:
    """Return (score_contribution, weight). Skips None."""
    if value is None:
        return 0.0, 0.0
    v = value
    w = warn
    f = flag
    if not higher_is_risk:
        v = -v
        w = -w
        f = -f
    if v <= w:
        sc = 0.0
    elif v >= f:
        sc = 100.0
    else:
        sc = (v - w) / max(f - w, 1e-9) * 100.0
    sub[key] = round(sc)
    return sc, 1.0


def safe_mean(x: List[float]) -> Optional[float]:
    x = [v for v in x if v is not None]
    return (sum(x)/len(x)) if x else None

# ---------------- HTTP Helpers ----------------


async def fetch_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, tries=3, timeout=15.0) -> Any:
    params = params or {}
    headers = headers or {}
    for i in range(tries):
        try:
            r = await client.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if i == tries - 1:
                log.warning("GET %s failed after %s tries: %s", url, tries, e)
                raise
            await asyncio.sleep(1.5*(i+1))


async def _safe(coro, default, label: str):
    try:
        return await coro
    except Exception as e:
        log.warning("%s failed: %s", label, e)
        return default

# ---------------- DataClient ----------------


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            headers={"User-Agent": "cycle-bot/1.0"})

    async def close(self):
        await self.client.aclose()

    # --- Coingecko / CMC global dominance ---
    async def dominance_from_cg(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        data = await fetch_json(self.client, "https://api.coingecko.com/api/v3/global")
        mcp = data.get("data", {}).get("market_cap_percentage", {})
        btc_dom = mcp.get("btc")
        eth_dom = mcp.get("eth")
        total_mcap = data.get("data", {}).get(
            "total_market_cap", {}).get("usd")
        return (float(btc_dom) if btc_dom is not None else None,
                float(eth_dom) if eth_dom is not None else None,
                float(total_mcap) if total_mcap is not None else None)

    async def dominance_from_cmc(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
        data = await fetch_json(self.client, "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest", headers=headers)
        d = data.get("data", {})
        btc_dom = d.get("btc_dominance")  # percent
        total_mcap = d.get("quote", {}).get("USD", {}).get("total_market_cap")
        return (float(btc_dom)/100.0 if btc_dom is not None else None, None, float(total_mcap) if total_mcap is not None else None)

    # --- OKX market data ---
    async def okx_candles(self, instId: str, bar: str, limit: int = 400) -> List[List[str]]:
        # OKX returns newest first
        j = await fetch_json(self.client, "https://www.okx.com/api/v5/market/candles", params={"instId": instId, "bar": bar, "limit": str(limit)})
        return list(reversed(j.get("data", [])))

    async def okx_weekly_closes(self, instId: str, limit: int = 400) -> List[float]:
        rows = await self.okx_candles(instId, "1W", limit)
        return [float(r[4]) for r in rows if len(r) > 4]

    async def okx_biweekly_closes(self, instId: str, limit: int = 200) -> List[float]:
        rows = await _safe(self.okx_candles(instId, "2W", limit), [], f"okx_2w_{instId}")
        if rows:
            return [float(r[4]) for r in rows if len(r) > 4]
        # fallback stitch from 1W
        wk = await _safe(self.okx_weekly_closes(instId, limit=limit*2), [], f"okx_1w_{instId}")
        if len(wk) < 2:
            return []
        out = []
        start = 1 if (len(wk) % 2 == 0) else 0
        for i in range(start, len(wk), 2):
            out.append(wk[i])
        return out

    async def okx_daily_closes(self, instId: str, limit: int = 500) -> List[float]:
        rows = await self.okx_candles(instId, "1D", limit)
        return [float(r[4]) for r in rows if len(r) > 4]

    async def okx_ticker_last(self, instId: str) -> Optional[float]:
        j = await fetch_json(self.client, "https://www.okx.com/api/v5/market/ticker", params={"instId": instId})
        d = j.get("data", [])
        return float(d[0]["last"]) if d else None

    async def okx_funding(self, instId: str) -> Optional[float]:
        j = await fetch_json(self.client, "https://www.okx.com/api/v5/public/funding-rate", params={"instId": instId})
        d = j.get("data", [])
        return float(d[0]["fundingRate"]) if d else None  # per 8h

    async def okx_open_interest_usd(self, instId: str) -> Optional[float]:
        # Prefer oiCcy * last price
        j = await fetch_json(self.client, "https://www.okx.com/api/v5/public/open-interest", params={"instType": "SWAP", "instId": instId})
        d = j.get("data", [])
        if not d:
            return None
        oi_ccy = float(d[0].get("oiCcy")) if d[0].get(
            "oiCcy") not in (None, "") else None
        last = await _safe(self.okx_ticker_last(instId), None, f"okx_last_{instId}")
        if oi_ccy is not None and last is not None:
            return oi_ccy * last
        # fallback rough
        oi = float(d[0].get("oi")) if d[0].get(
            "oi") not in (None, "") else None
        if oi is not None and last is not None:
            return oi * last
        return None

# ---------------- Metrics ----------------


def thresholds_for_profile(profile: str) -> Dict[str, float]:
    mult = PROFILE_MULTIPLIER.get(profile, 1.0)
    out = {}
    for k, v in BASE_THRESHOLDS.items():
        # multiply only values that are not percents in [0,1]? We keep it simple: multiply all numeric thresholds.
        out[k] = v * mult
    # For percentages that are “≤ proximity”, we keep as-is (multiplying is fine here: aggressive => looser).
    return out


async def google_trends_avg7() -> Optional[float]:
    try:
        pytrends = TrendReq(hl="en-US", tz=0)
        kw = ["crypto", "bitcoin", "ethereum"]
        pytrends.build_payload(kw, timeframe="now 7-d", geo="")
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            return None
        # average across columns and days
        vals = df[kw].mean(axis=1).tolist()
        return float(sum(vals) / len(vals)) if vals else None
    except Exception as e:
        log.warning("google_trends_avg7 failed: %s", e)
        return None


async def fear_greed() -> Dict[str, Optional[float]]:
    # today
    j1 = await fetch_json(httpx.AsyncClient(), "https://api.alternative.me/fng/", params={"limit": "1"})
    v_today = float(j1["data"][0]["value"])
    # last 60 for MA and persistence
    j2 = await fetch_json(httpx.AsyncClient(), "https://api.alternative.me/fng/", params={"limit": "60"})
    arr = [float(x["value"])
           for x in j2.get("data", [])][::-1]  # oldest -> newest
    ma14 = safe_mean(arr[-14:]) if len(arr) >= 14 else None
    ma30 = safe_mean(arr[-30:]) if len(arr) >= 30 else None
    # greed days in a row (≥70)
    days_row = 0
    for x in reversed(arr):
        if x >= 70:
            days_row += 1
        else:
            break
    pct30 = (sum(1 for x in arr[-30:] if x >= 70) /
             30.0) if len(arr) >= 30 else None
    return {"today": v_today, "ma14": ma14, "ma30": ma30, "days_row": days_row, "pct30": pct30}


async def pi_cycle_ratio_from_okx(dc: DataClient) -> Optional[float]:
    """111DMA / (2 * 350DMA) from OKX BTC-USDT daily closes."""
    try:
        closes = await dc.okx_daily_closes("BTC-USDT", 500)
        if len(closes) < 350:
            return None
        ma111 = sma(closes, 111)
        ma350 = sma(closes, 350)
        if not ma111 or not ma350 or ma350 <= 0:
            return None
        ratio = ma111 / (2.0 * ma350)
        if ratio <= 0 or ratio > 5:
            return None
        return ratio
    except Exception as e:
        log.warning("PiCycle from OKX failed: %s", e)
        return None


def rsi_block(arr: List[float], rsi_len=14, ma_len=9) -> Optional[Dict[str, float]]:
    if len(arr) < (rsi_len + ma_len + 5):
        return None
    r = rsi(arr, rsi_len)
    if len(r) < ma_len + 2:
        return None
    ma = [sum(r[i-ma_len:i])/ma_len for i in range(ma_len, len(r)+1)]
    r_last = r[-1]
    ma_last = ma[-1]
    losing = (len(r) >= 2 and len(ma) >=
              2 and r[-2] >= ma[-2] and r_last < ma_last)
    return {"rsi": r_last, "rsi_ma": ma_last, "losing_ma": 1.0 if losing else 0.0}


def fib_proximity_1272(series: List[float], lookback: int = 100) -> Optional[float]:
    """Return absolute proximity to 1.272 extension based on lookback high/low, as a fraction (e.g., 0.1937 = 19.37%)."""
    if len(series) < lookback + 2:
        return None
    sub = series[-lookback:]
    lo = min(sub)
    hi = max(sub)
    if hi <= lo:
        return None
    ext = hi + 0.272*(hi - lo)
    last = sub[-1]
    if last <= 0:
        return None
    return abs(ext - last)/last


async def gather_metrics(dc: DataClient, profile: str) -> Dict[str, Any]:
    thr = thresholds_for_profile(profile)

    # --- Dominance & market caps ---
    if CMC_API_KEY:
        btc_dom, eth_dom, total_mcap = await _safe(dc.dominance_from_cmc(), (None, None, None), "dominance_cmc")
    else:
        btc_dom, eth_dom, total_mcap = await _safe(dc.dominance_from_cg(), (None, None, None), "dominance_cg")

    altcap_btc_ratio = None
    if btc_dom is not None and btc_dom > 0:
        altcap_btc_ratio = (1.0 - btc_dom) / btc_dom

    # ETH/BTC weekly close
    ethbtc_w = await _safe(dc.okx_weekly_closes("ETH-BTC", 400), [], "ethbtc_1w")
    eth_btc_ratio = ethbtc_w[-1] if ethbtc_w else None

    # --- Derivatives: OKX funding basket, OI BTC/ETH ---
    # funding (per 8h)
    fund_syms = {
        "BTCUSDT": "BTC-USDT-SWAP",
        "ETHUSDT": "ETH-USDT-SWAP",
        "SOLUSDT": "SOL-USDT-SWAP",
        "XRPUSDT": "XRP-USDT-SWAP",
        "DOGEUSDT": "DOGE-USDT-SWAP",
    }
    funding_vals = {}
    for k, inst in fund_syms.items():
        funding_vals[k] = await _safe(dc.okx_funding(inst), None, f"fund_{inst}")
    # convert per-8h to percent
    funding_pcts = {k: (v*100.0 if v is not None else None)
                    for k, v in funding_vals.items()}
    funding_list_abs = [abs(v) for v in funding_vals.values() if v is not None]
    funding_max_abs = max(funding_list_abs) if funding_list_abs else None
    funding_median = None
    arrp = sorted([v for v in funding_vals.values() if v is not None])
    if arrp:
        mid = len(arrp)//2
        funding_median = (arrp[mid] if len(arrp) %
                          2 == 1 else (arrp[mid-1]+arrp[mid])/2.0) * 100.0

    # OI in USD
    oi_btc = await _safe(dc.okx_open_interest_usd("BTC-USDT-SWAP"), None, "oi_btc")
    oi_eth = await _safe(dc.okx_open_interest_usd("ETH-USDT-SWAP"), None, "oi_eth")

    # --- Sentiment ---
    trends7 = await _safe(google_trends_avg7(), None, "google_trends")
    fng = await _safe(fear_greed(), {"today": None, "ma14": None, "ma30": None, "days_row": None, "pct30": None}, "fng")

    # --- Cycle (Pi) ---
    pi_ratio = await _safe(pi_cycle_ratio_from_okx(dc), None, "pi_cycle_okx")

    # --- Momentum (2W) & Extensions (1W) ---
    btc_2w = await _safe(dc.okx_biweekly_closes("BTC-USDT", 200), [], "btc_2w")
    ethbtc_2w = await _safe(dc.okx_biweekly_closes("ETH-BTC", 200), [], "ethbtc_2w")

    # alt equal-weight 2W index (normalize each)
    alt_2w_lines: List[List[float]] = []
    for s in ALT_BASKET:
        alt_2w_lines.append(await _safe(dc.okx_biweekly_closes(s, 200), [], f"{s}_2w"))
    alt_ix_2w: List[float] = []
    normd = []
    for arr in alt_2w_lines:
        if arr:
            base = arr[0]
            if base and base > 0:
                normd.append([v/base for v in arr])
    if normd:
        L = min(len(a) for a in normd)
        for i in range(L):
            alt_ix_2w.append(sum(a[i] for a in normd)/len(normd))

    # RSI/MA blocks
    rsi_ma_len = 9
    btc_rsi2w = rsi_block(btc_2w, 14, rsi_ma_len)
    ethbtc_rsi2w = rsi_block(ethbtc_2w, 14, rsi_ma_len)
    alt_rsi2w = rsi_block(alt_ix_2w, 14, rsi_ma_len)

    # StochRSI
    btc_k, btc_d = stoch_rsi(btc_2w, 14, 3, 3)
    alt_k, alt_d = stoch_rsi(alt_ix_2w, 14, 3, 3)
    btc_stoch = {"k": (btc_k[-1] if btc_k else None),
                 "d": (btc_d[-1] if btc_d else None)}
    alt_stoch = {"k": (alt_k[-1] if alt_k else None),
                 "d": (alt_d[-1] if alt_d else None)}

    # Fibonacci proximity (1W) for BTC and for alt equal-weight (1W)
    btc_1w = await _safe(dc.okx_weekly_closes("BTC-USDT", 400), [], "btc_1w")
    fib_btc = fib_proximity_1272(btc_1w, 100)
    # build alt 1W equal-weight
    alt_1w_lines: List[List[float]] = []
    for s in ALT_BASKET:
        alt_1w_lines.append(await _safe(dc.okx_weekly_closes(s, 400), [], f"{s}_1w"))
    alt_ix_1w: List[float] = []
    normd1 = []
    for arr in alt_1w_lines:
        if arr:
            base = arr[0]
            if base and base > 0:
                normd1.append([v/base for v in arr])
    if normd1:
        L = min(len(a) for a in normd1)
        for i in range(L):
            alt_ix_1w.append(sum(a[i] for a in normd1)/len(normd1))
    fib_alt = fib_proximity_1272(alt_ix_1w, 100)

    # Package
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "btc_dom": btc_dom,
        "eth_btc_ratio": eth_btc_ratio,
        "altcap_btc_ratio": altcap_btc_ratio,
        "funding_pcts": funding_pcts,      # dict per symbol (percent per 8h)
        "funding_max_abs_pct": (max(abs(v) for v in funding_pcts.values() if v is not None) if any(v is not None for v in funding_pcts.values()) else None),
        "funding_median_pct": funding_median,
        "funding_top3": sorted([(k, abs(v)) for k, v in funding_vals.items() if v is not None], key=lambda x: x[1], reverse=True)[:3],
        "oi_btc_usd": oi_btc,
        "oi_eth_usd": oi_eth,
        "trends7": trends7,
        "fng": fng,
        "pi_ratio": pi_ratio,  # e.g., 0.97 = 97% of cross
        "mom2w": {
            "btc": btc_rsi2w, "ethbtc": ethbtc_rsi2w, "alt": alt_rsi2w,
            "btc_stoch": btc_stoch, "alt_stoch": alt_stoch
        },
        "fib": {"btc": fib_btc, "alt": fib_alt},
        "thr": thr,
    }


def composite_score(metrics: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    thr = metrics["thr"]
    subs: Dict[str, int] = {}
    num = 0.0
    den = 0.0

    # Market structure
    n, w = add_scaled(subs, "altcap_vs_btc", metrics.get(
        "altcap_btc_ratio"), thr["altcap_btc_ratio_warn"], thr["altcap_btc_ratio_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "eth_btc", metrics.get(
        "eth_btc_ratio"), thr["eth_btc_ratio_warn"], thr["eth_btc_ratio_flag"], True)
    num += n
    den += w

    # Derivatives
    n, w = add_scaled(subs, "funding", (metrics.get("funding_max_abs_pct")/100.0 if metrics.get("funding_max_abs_pct") is not None else None),
                      thr["funding_rate_abs_warn"], thr["funding_rate_abs_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "OI_BTC", metrics.get("oi_btc_usd"),
                      thr["oi_btc_usd_warn"], thr["oi_btc_usd_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "OI_ETH", metrics.get("oi_eth_usd"),
                      thr["oi_eth_usd_warn"], thr["oi_eth_usd_flag"], True)
    num += n
    den += w

    # Sentiment
    n, w = add_scaled(subs, "trends", metrics.get(
        "trends7"), thr["google_trends_7d_warn"], thr["google_trends_7d_flag"], True)
    num += n
    den += w
    fng = metrics.get("fng") or {}
    n, w = add_scaled(subs, "F&G", fng.get("today"),
                      thr["fear_greed_warn"], thr["fear_greed_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "F&G14", fng.get(
        "ma14"), thr["fear_greed_ma14_warn"], thr["fear_greed_ma14_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "F&G30", fng.get(
        "ma30"), thr["fear_greed_ma30_warn"], thr["fear_greed_ma30_flag"], True)
    num += n
    den += w
    # Greed persistence (use max of days-row progress and pct30 progress)
    persist = None
    if fng.get("days_row") is not None:
        dr = fng["days_row"]
        w0 = thr["greed_days_row_warn"]
        f0 = thr["greed_days_row_flag"]
        persist = 0.0 if dr <= w0 else (
            100.0 if dr >= f0 else (dr - w0)/max(f0 - w0, 1e-9)*100.0)
    if fng.get("pct30") is not None:
        p = fng["pct30"]
        w0 = thr["greed_pct30_warn"]
        f0 = thr["greed_pct30_flag"]
        psc = 0.0 if p <= w0 else (
            100.0 if p >= f0 else (p - w0)/max(f0 - w0, 1e-9)*100.0)
        persist = psc if persist is None else max(persist, psc)
    if persist is not None:
        subs["F&GPersist"] = round(persist)
        num += persist
        den += 1.0

    # Cycle
    pi = metrics.get("pi_ratio")
    n, w = add_scaled(subs, "Pi", (pi*100.0 if pi is not None else None),
                      thr["pi_cycle_ratio_warn"]*100.0, thr["pi_cycle_ratio_flag"]*100.0, True)
    num += n
    den += w

    # Momentum (2W RSI)
    m2w = metrics.get("mom2w", {})
    for label, key, warnk, flagk in [
        ("RSI_BTC_2W", "btc", "btc_2w_rsi_warn", "btc_2w_rsi_flag"),
        ("RSI_ETHBTC_2W", "ethbtc", "ethbtc_2w_rsi_warn", "ethbtc_2w_rsi_flag"),
        ("RSI_ALT_2W", "alt", "alt_2w_rsi_warn", "alt_2w_rsi_flag"),
    ]:
        block = m2w.get(key)
        val = block.get("rsi") if block else None
        n, w = add_scaled(subs, label, val, thr[warnk], thr[flagk], True)
        num += n
        den += w

    # Fibonacci proximity (inverse: nearer => more risk)
    fib = metrics.get("fib", {})
    n, w = add_scaled(subs, "Fib_BTC", (1.0 - min(1.0, (fib.get("btc") or 1.0))),
                      1.0 - thr["fib_prox_warn"], 1.0 - thr["fib_prox_flag"], True)
    num += n
    den += w
    n, w = add_scaled(subs, "Fib_ALT", (1.0 - min(1.0, (fib.get("alt") or 1.0))),
                      1.0 - thr["fib_prox_warn"], 1.0 - thr["fib_prox_flag"], True)
    num += n
    den += w

    comp = round(num/den) if den > 0 else 0
    return comp, subs


def tri_color(value: int, yel=40, red_=70) -> str:
    if value >= red_:
        return RED
    if value >= yel:
        return YELLOW
    return GREEN


# ---------------- Telegram ----------------


def allowed(update: Update) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    chat_id = update.effective_chat.id if update.effective_chat else None
    return (str(chat_id) in ALLOWED_CHAT_IDS)


def profile_for_chat(chat_id: int) -> str:
    return CHAT_PROFILE.get(chat_id, DEFAULT_PROFILE)


def set_profile(chat_id: int, p: str):
    p = p.lower().strip()
    if p not in PROFILE_MULTIPLIER:
        p = DEFAULT_PROFILE
    CHAT_PROFILE[chat_id] = p


def fmt_metrics(m: Dict[str, Any]) -> str:
    thr = m["thr"]
    profile = m["profile"]

    # Market structure badges
    b_btc_dom = badge(
        m.get("btc_dom"), thr["btc_dominance_warn"], thr["btc_dominance_flag"], True)
    b_ethbtc = badge(m.get("eth_btc_ratio"),
                     thr["eth_btc_ratio_warn"], thr["eth_btc_ratio_flag"], True)
    b_altbtc = badge(m.get("altcap_btc_ratio"),
                     thr["altcap_btc_ratio_warn"], thr["altcap_btc_ratio_flag"], True)

    # Derivatives
    fund_max = m.get("funding_max_abs_pct")            # percent
    fund_med = m.get("funding_median_pct")             # percent
    b_fmax = badge((fund_max/100.0 if fund_max is not None else None),
                   thr["funding_rate_abs_warn"], thr["funding_rate_abs_flag"], True)
    b_oi_btc = badge(m.get("oi_btc_usd"),
                     thr["oi_btc_usd_warn"], thr["oi_btc_usd_flag"], True)
    b_oi_eth = badge(m.get("oi_eth_usd"),
                     thr["oi_eth_usd_warn"], thr["oi_eth_usd_flag"], True)

    # Sentiment
    b_tr = badge(m.get("trends7"),
                 thr["google_trends_7d_warn"], thr["google_trends_7d_flag"], True)
    fng = m.get("fng") or {}
    b_fng = badge(fng.get("today"),
                  thr["fear_greed_warn"], thr["fear_greed_flag"], True)
    b_fng14 = badge(
        fng.get("ma14"), thr["fear_greed_ma14_warn"], thr["fear_greed_ma14_flag"], True)
    b_fng30 = badge(
        fng.get("ma30"), thr["fear_greed_ma30_warn"], thr["fear_greed_ma30_flag"], True)

    # Greed persistence descriptor
    gp_desc = "n/a"
    if fng.get("days_row") is not None or fng.get("pct30") is not None:
        dr = fng.get("days_row")
        pct30 = fng.get("pct30")
        gp_desc = f"{(dr or 0)} days in a row"
        if pct30 is not None:
            gp_desc += f" | {int(round(pct30*100))}% of last 30 days ≥ 70"

    # Cycle
    pi = m.get("pi_ratio")
    pi_pct = (f"{pi*100:.1f}%" if pi is not None else "n/a")
    b_pi = badge((pi*100.0 if pi is not None else None),
                 thr["pi_cycle_ratio_warn"]*100.0, thr["pi_cycle_ratio_flag"]*100.0, True)

    # Momentum (2W)
    def fmt_rsi_line(label, block, warn, flag):
        if not block:
            return f"• {label} RSI (2W): {GREEN}n/a"
        r = block["rsi"]
        rma = block["rsi_ma"]
        losing = bool(block["losing_ma"])
        color = RED if r >= flag else (YELLOW if r >= warn else GREEN)
        extra = " — ⚠ losing RSI MA" if losing else ""
        return (f"• {label} RSI (2W): {color}{r:.1f} (MA {rma:.1f}){extra} (warn ≥ {warn:.1f}, flag ≥ {flag:.1f})")

    mom = m.get("mom2w", {})
    line_btc = fmt_rsi_line("BTC", mom.get(
        "btc"), thr["btc_2w_rsi_warn"], thr["btc_2w_rsi_flag"])
    line_eth = fmt_rsi_line("ETH/BTC", mom.get("ethbtc"),
                            thr["ethbtc_2w_rsi_warn"], thr["ethbtc_2w_rsi_flag"])
    line_alt = fmt_rsi_line("ALT basket (equal-weight)", mom.get("alt"),
                            thr["alt_2w_rsi_warn"], thr["alt_2w_rsi_flag"])

    # Stoch (2W) shown compact
    def fmt_stoch(label, obj):
        k = obj.get("k") if obj else None
        d = obj.get("d") if obj else None
        if k is None or d is None:
            return f"• {label} Stoch RSI (2W) K/D: {GREEN}n/a"
        ob = (max(k, d) >= 0.80)
        cross_dn = (k < d and ob)
        color = RED if cross_dn else (YELLOW if ob else GREEN)
        return f"• {label} Stoch RSI (2W) K/D: {color}{k:.2f}/{d:.2f} (overbought ≥ 0.80; red = bearish cross from OB)"

    stoch_btc = fmt_stoch("BTC", mom.get("btc_stoch"))
    stoch_alt = fmt_stoch("ALT basket", mom.get("alt_stoch"))

    # Fibonacci proximity (1W)
    fib_btc = m.get("fib", {}).get("btc")
    fib_alt = m.get("fib", {}).get("alt")
    b_fib_btc = badge((1.0 - min(1.0, fib_btc or 1.0)), 1.0 -
                      thr["fib_prox_warn"], 1.0 - thr["fib_prox_flag"], True)
    b_fib_alt = badge((1.0 - min(1.0, fib_alt or 1.0)), 1.0 -
                      thr["fib_prox_warn"], 1.0 - thr["fib_prox_flag"], True)

    # Composite + Top-5
    comp, subs = composite_score(m)
    comp_badge = tri_color(comp)
    # Top-5 contributors
    nice = {
        "altcap_vs_btc": "Alt mcap vs BTC",
        "eth_btc": "ETH/BTC strength",
        "funding": "Funding extremes",
        "OI_BTC": "BTC Open Interest",
        "OI_ETH": "ETH Open Interest",
        "trends": "Google Trends (7d)",
        "F&G": "Fear & Greed (today)",
        "F&G14": "Fear & Greed 14-d",
        "F&G30": "Fear & Greed 30-d",
        "F&GPersist": "Greed persistence",
        "Pi": "Pi Cycle proximity",
        "RSI_BTC_2W": "BTC RSI (2W)",
        "RSI_ETHBTC_2W": "ETH/BTC RSI (2W)",
        "RSI_ALT_2W": "ALT basket RSI (2W)",
        "Fib_BTC": "BTC Fib 1.272 (1W)",
        "Fib_ALT": "ALT Fib 1.272 (1W)",
    }
    top5 = sorted(subs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top5_lines = []
    for k, v in top5:
        nm = nice.get(k, k)
        # cosmetic for list: mild/yellow/red thresholds
        dot = tri_color(v, 50, 80)
        top5_lines.append(f"• {nm}: {dot}{v}/100")

    # funding top-3
    tops = m.get("funding_top3") or []
    tops_line = ""
    if tops:
        # values are per 8h (abs); print as percent
        def fp(x): return f"{x*100:.3f}%"
        tops_line = "Top-3 funding extremes: " + \
            ", ".join([f"{sym} {fp(val)}" for sym, val in tops])

    # Build text (HTML-safe: no '<' or '>' except tags we explicitly add)
    ts = m.get("ts", "")
    lines = []
    lines.append(f"📊 <b>Crypto Market Snapshot</b> — {ts} UTC")
    lines.append(f"Profile: {profile}\n")

    lines.append("<b>Market Structure</b>")
    lines.append(
        f"• Bitcoin market share of total crypto: {b_btc_dom}{pct(m.get('btc_dom'))}  (warn ≥ {pct(thr['btc_dominance_warn'])}, flag ≥ {pct(thr['btc_dominance_flag'])})")
    lines.append(f"• Ether price relative to Bitcoin (ETH/BTC): {b_ethbtc}{(m.get('eth_btc_ratio') if m.get('eth_btc_ratio') is not None else 'n/a'):.5f}" if m.get(
        'eth_btc_ratio') is not None else f"• Ether price relative to Bitcoin (ETH/BTC): {b_ethbtc}n/a")
    lines.append(f"• Altcoin market cap / Bitcoin market cap: {b_altbtc}{(m.get('altcap_btc_ratio') if m.get('altcap_btc_ratio') is not None else 'n/a') if m.get('altcap_btc_ratio') is None else f'{m.get('altcap_btc_ratio'):.2f}'}  (warn ≥ {BASE_THRESHOLDS['altcap_btc_ratio_warn']:.2f}, flag ≥ {BASE_THRESHOLDS['altcap_btc_ratio_flag']:.2f})")

    lines.append("\n<b>Derivatives</b>")
    fm = f"{fund_max:.3f}%" if fund_max is not None else "n/a"
    fmed = f"{fund_med:.3f}%" if fund_med is not None else "n/a"
    lines.append(
        f"• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — max: {b_fmax}{fm} | median: {b_fmax}{fmed}  (warn ≥ {BASE_THRESHOLDS['funding_rate_abs_warn']*100:.3f}%, flag ≥ {BASE_THRESHOLDS['funding_rate_abs_flag']*100:.3f}%)")
    if tops_line:
        lines.append(f"  {tops_line}")
    lines.append(
        f"• Bitcoin open interest (USD): {b_oi_btc}{usd(m.get('oi_btc_usd'))}  (warn ≥ {usd(BASE_THRESHOLDS['oi_btc_usd_warn'])}, flag ≥ {usd(BASE_THRESHOLDS['oi_btc_usd_flag'])})")
    lines.append(
        f"• Ether open interest (USD): {b_oi_eth}{usd(m.get('oi_eth_usd'))}  (warn ≥ {usd(BASE_THRESHOLDS['oi_eth_usd_warn'])}, flag ≥ {usd(BASE_THRESHOLDS['oi_eth_usd_flag'])})")

    lines.append("\n<b>Sentiment</b>")
    lines.append(f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {b_tr}{m.get('trends7'):.1f}" if m.get(
        "trends7") is not None else f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {b_tr}n/a")
    lines.append(
        f"• Fear & Greed Index (overall crypto): {b_fng}{int(fng.get('today')) if fng.get('today') is not None else 'n/a'} (Greed)  (warn ≥ {BASE_THRESHOLDS['fear_greed_warn']}, flag ≥ {BASE_THRESHOLDS['fear_greed_flag']})")
    lines.append(f"• Fear & Greed 14-day average: {b_fng14}{f'{fng.get('ma14'):.1f}' if fng.get('ma14') is not None else 'n/a'}  (warn ≥ {BASE_THRESHOLDS['fear_greed_ma14_warn']}, flag ≥ {BASE_THRESHOLDS['fear_greed_ma14_flag']})")
    lines.append(f"• Fear & Greed 30-day average: {b_fng30}{f'{fng.get('ma30'):.1f}' if fng.get('ma30') is not None else 'n/a'}  (warn ≥ {BASE_THRESHOLDS['fear_greed_ma30_warn']}, flag ≥ {BASE_THRESHOLDS['fear_greed_ma30_flag']})")
    lines.append(
        f"• Greed persistence: {tri_color(subs.get('F&GPersist', 0), 33, 66)}{gp_desc}  (warn: days ≥ {BASE_THRESHOLDS['greed_days_row_warn']} or pct ≥ {int(BASE_THRESHOLDS['greed_pct30_warn']*100)}%; flag: days ≥ {BASE_THRESHOLDS['greed_days_row_flag']} or pct ≥ {int(BASE_THRESHOLDS['greed_pct30_flag']*100)}%)")

    lines.append("\n<b>Cycle & On-Chain</b>")
    lines.append(
        f"• Pi Cycle Top proximity: {b_pi}{pi_pct} of trigger (100% = cross)")

    lines.append("\n<b>Momentum (2W) & Extensions (1W)</b>")
    lines.append(line_btc)
    lines.append(line_eth)
    lines.append(line_alt)
    lines.append(stoch_btc)
    lines.append(stoch_alt)
    if fib_btc is not None:
        lines.append(
            f"• BTC Fibonacci extension proximity: {b_fib_btc}1.272 @ {(fib_btc*100):.2f}% away (warn ≤ {BASE_THRESHOLDS['fib_prox_warn']*100:.1f}%, flag ≤ {BASE_THRESHOLDS['fib_prox_flag']*100:.1f}%)")
    else:
        lines.append(f"• BTC Fibonacci extension proximity: {b_fib_btc}n/a")
    if fib_alt is not None:
        lines.append(
            f"• ALT basket Fibonacci proximity: {b_fib_alt}1.272 @ {(fib_alt*100):.2f}% away (warn ≤ {BASE_THRESHOLDS['fib_prox_warn']*100:.1f}%, flag ≤ {BASE_THRESHOLDS['fib_prox_flag']*100:.1f}%)")
    else:
        lines.append(f"• ALT basket Fibonacci proximity: {b_fib_alt}n/a")

    lines.append("\n<b>Alt-Top Certainty (Composite)</b>")
    lines.append(
        f"• Certainty: {comp_badge}{comp}/100 (yellow ≥ 40, red ≥ 70)")
    if top5_lines:
        lines.append("• Top drivers:")
        lines.extend(top5_lines)

    return "\n".join(lines)

# ---------------- Bot Handlers ----------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    set_profile(chat_id, profile_for_chat(chat_id))  # ensure initialized
    msg = (
        "Hi! I track cycle-top risk for the crypto market.\n\n"
        "Commands:\n"
        "/set_profile <conservative|moderate|aggressive>\n"
        "/status  — fetch current assessment\n"
        "/assess  — same as /status\n\n"
        "You’ll also receive a once-a-day summary and 15-minute alerts if flags trigger."
    )
    await update.message.reply_text(msg)


async def cmd_set_profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    args = context.args or []
    if not args or args[0].lower() not in PROFILE_MULTIPLIER:
        await update.message.reply_text("Usage: /set_profile conservative|moderate|aggressive")
        return
    set_profile(chat_id, args[0].lower())
    await update.message.reply_text(f"Profile set to {args[0].lower()}.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    chat_id = update.effective_chat.id
    profile = profile_for_chat(chat_id)
    dc = DataClient()
    try:
        m = await gather_metrics(dc, profile)
        text = fmt_metrics(m)
        await update.message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        log.exception("status failed")
        await update.message.reply_text(f"⚠️ Could not fetch metrics right now: {e}")
    finally:
        await dc.close()

# Alias


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cmd_status(update, context)

# ---------------- Scheduled jobs ----------------


async def push_summary(app: Application):
    # If ALLOWED_CHAT_IDS set, broadcast to those; else no-op unless you want a default.
    targets = ALLOWED_CHAT_IDS or set()
    if not targets:
        return
    for tid in targets:
        chat_id = int(tid)
        profile = profile_for_chat(chat_id)
        dc = DataClient()
        try:
            m = await gather_metrics(dc, profile)
            await app.bot.send_message(chat_id=chat_id, text=fmt_metrics(m), parse_mode="HTML")
        except Exception as e:
            log.warning("push_summary to %s failed: %s", chat_id, e)
        finally:
            await dc.close()


async def push_alerts(app: Application):
    # Lightweight alerting: if composite goes red (≥70) or new flags appear, send snapshot
    targets = ALLOWED_CHAT_IDS or set()
    if not targets:
        return
    for tid in targets:
        chat_id = int(tid)
        profile = profile_for_chat(chat_id)
        dc = DataClient()
        try:
            m = await gather_metrics(dc, profile)
            comp, subs = composite_score(m)
            if comp >= 70:
                await app.bot.send_message(chat_id=chat_id, text=fmt_metrics(m), parse_mode="HTML")
        except Exception as e:
            log.warning("push_alerts to %s failed: %s", chat_id, e)
        finally:
            await dc.close()

# ---------------- Health server (Koyeb) ----------------


async def handle_health(request):
    return web.json_response({"ok": True, "ts": datetime.now(timezone.utc).isoformat()})


async def handle_root(request):
    return web.Response(text="cycle-bot up")

# ---------------- Main ----------------


async def main():
    log.info("Loaded configuration from environment")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("set_profile", cmd_set_profile))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("assess", cmd_assess))

    # Health server
    aio = web.Application()
    aio.add_routes([web.get("/", handle_root),
                   web.get("/health", handle_health)])
    runner = web.AppRunner(aio)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=8080)
    await site.start()
    log.info("Health server listening on :8080")

    # Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(lambda: asyncio.create_task(
        push_summary(app)), CronTrigger.from_crontab(DAILY_SUMMARY_CRON))
    scheduler.add_job(lambda: asyncio.create_task(
        push_alerts(app)), CronTrigger.from_crontab(ALERTS_CRON))
    scheduler.start()

    log.info("Bot running. Press Ctrl+C to exit.")
    await app.initialize()
    await app.bot.delete_webhook(drop_pending_updates=True)
    await app.start()
    await app.updater.start_polling()
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
