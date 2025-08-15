from aiohttp import web
import os
import asyncio
import logging
import json
from datetime import datetime, timezone, timedelta

import httpx
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes
)

import math
import statistics

# Optional heavy deps (pytrends/pandas/numpy) are available per requirements.txt
from pytrends.request import TrendReq

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("crypto-cycle-bot")

# ---------- Config ----------
DEFAULT_CFG = {
    "risk_profile": os.getenv("RISK_PROFILE", "moderate"),
    "daily_push_utc_hhmm": "14:00",     # once-a-day update (UTC)
    "alerts_every_minutes": 15,         # intraday checks
    "funding_symbols": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"],
    "alt_basket": ["SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT", "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"],
    "telegram": {
        "token": os.getenv("TELEGRAM_TOKEN", ""),
        "admin_chat_id": os.getenv("ADMIN_CHAT_ID", "")  # for scheduled pushes
    },
}


def load_config():
    cfg_path = os.path.join(os.getcwd(), "config.yml")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            log.warning("Failed to read config.yml: %s", e)
    # merge defaults

    def deepmerge(a, b):
        for k, v in b.items():
            if isinstance(v, dict):
                a[k] = deepmerge(a.get(k, {}) if isinstance(
                    a.get(k), dict) else {}, v)
            else:
                a[k] = a.get(k, v) if a.get(k) not in (None, "") else v
        return a
    merged = deepmerge(cfg, DEFAULT_CFG)
    return merged


CFG = load_config()
if CFG["telegram"]["token"]:
    log.info("Loaded configuration from config.yml")
else:
    log.warning("TELEGRAM_TOKEN missing. Set env or config.yml.")

# ---------- Small utils ----------


def pct(x):
    try:
        return f"{x:.2f}%"
    except Exception:
        return "n/a"


def usd(x):
    try:
        return f"${x:,.0f}"
    except Exception:
        return "n/a"


def green_yellow_red(val, warn, flag, reverse=False):
    """3-color traffic light. reverse=True makes lower=red, higher=green."""
    if val is None:
        return "🟡"
    if reverse:
        # lower worse
        if val <= flag:
            return "🔴"
        if val <= warn:
            return "🟡"
        return "🟢"
    else:
        if val >= flag:
            return "🔴"
        if val >= warn:
            return "🟡"
        return "🟢"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------- HTTP/Data client ----------


class DataClient:
    def __init__(self, timeout=15.0):
        self.client = httpx.AsyncClient(timeout=timeout, headers={
                                        "User-Agent": "cc-bot/1.0"})
        self.trends = TrendReq(hl="en-US", tz=0)
        self.cmc_key = os.getenv("CMC_API_KEY", "").strip()

    async def close(self):
        await self.client.aclose()

    # ---------- Coingecko (global + simple price) ----------
    async def coingecko_global(self):
        try:
            r = await self.client.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            return r.json().get("data", {})
        except Exception as e:
            log.warning("coingecko_global failed: %s", e)
            return {}

    async def cg_simple_price(self, ids=("bitcoin", "ethereum"), vs="usd"):
        try:
            params = {"ids": ",".join(ids), "vs_currencies": vs}
            r = await self.client.get("https://api.coingecko.com/api/v3/simple/price", params=params)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("cg_simple_price failed: %s", e)
            return {}

    # ---------- OKX funding, ticker, OI, candles ----------
    async def okx_funding_rate(self, inst_id):
        try:
            r = await self.client.get("https://www.okx.com/api/v5/public/funding-rate", params={"instId": inst_id})
            r.raise_for_status()
            data = r.json().get("data") or []
            if not data:
                return None
            fr = safe_float(data[0].get("fundingRate"))
            return fr  # per 8h (decimal), e.g. 0.0001 = 0.01%
        except Exception as e:
            log.warning("okx_funding_rate %s failed: %s", inst_id, e)
            return None

    async def okx_ticker_last(self, inst_id):
        try:
            r = await self.client.get("https://www.okx.com/api/v5/market/ticker", params={"instId": inst_id})
            r.raise_for_status()
            data = r.json().get("data") or []
            if not data:
                return None
            last = safe_float(data[0].get("last"))
            return last
        except Exception as e:
            log.warning("okx_ticker_last %s failed: %s", inst_id, e)
            return None

    async def okx_open_interest_usd(self, inst_id):
        """Approx USD OI = oiCcy * last_price (USDT-margined swap)."""
        try:
            oi_r = await self.client.get("https://www.okx.com/api/v5/public/open-interest",
                                         params={"instType": "SWAP", "instId": inst_id})
            oi_r.raise_for_status()
            oi_data = oi_r.json().get("data") or []
            if not oi_data:
                return None
            oi_ccy = safe_float(oi_data[0].get("oiCcy"))  # in coin units
            last = await self.okx_ticker_last(inst_id)
            if oi_ccy is None or last is None:
                return None
            return oi_ccy * last
        except Exception as e:
            log.warning("okx_open_interest_usd %s failed: %s", inst_id, e)
            return None

    async def okx_candles(self, inst_id, bar="1W", limit=400):
        """Return list of closes (oldest->newest) for OKX instrument."""
        try:
            r = await self.client.get("https://www.okx.com/api/v5/market/candles",
                                      params={"instId": inst_id, "bar": bar, "limit": str(limit)})
            r.raise_for_status()
            raw = r.json().get("data") or []
            closes_desc = [safe_float(row[4])
                           for row in raw if safe_float(row[4]) is not None]
            closes = list(reversed(closes_desc))
            return closes
        except Exception as e:
            log.warning("okx_candles %s %s failed: %s", inst_id, bar, e)
            return []

    async def okx_btc_daily_closes(self, limit=500):
        return await self.okx_candles("BTC-USDT", bar="1D", limit=limit)

    # ---------- Google Trends ----------
    def google_trends_score(self, kw_list=("crypto", "bitcoin", "ethereum"), days=7):
        try:
            self.trends.build_payload(
                list(kw_list), timeframe=f"today {days}-d", geo="")
            df = self.trends.interest_over_time()
            if df is None or df.empty:
                return None
            # mean of last window across series
            cols = [c for c in df.columns if c != "isPartial"]
            vals = []
            for c in cols:
                vals.extend(df[c].tail(days).tolist())
            vals = [v for v in vals if v is not False]
            if not vals:
                return None
            return sum(vals) / len(vals)
        except Exception as e:
            log.warning("google_trends_score failed: %s", e)
            return None

    # ---------- Fear & Greed ----------
    async def fear_greed(self, limit=1):
        try:
            r = await self.client.get("https://api.alternative.me/fng/", params={"limit": str(limit)})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            log.warning("fear_greed failed: %s", e)
            return {}

    # ---------- CMC TOTAL3 (optional) ----------
    async def cmc_total3(self):
        """Return TOTAL3 market cap (USD) if CMC_API_KEY provided, else None."""
        if not self.cmc_key:
            return None
        try:
            # CMC Global Metrics v1:
            # We'll approximate TOTAL3 by total_market_cap - (BTC+ETH mcap)
            headers = {"X-CMC_PRO_API_KEY": self.cmc_key}
            r = await self.client.get("https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                                      headers=headers, params={"convert": "USD"})
            r.raise_for_status()
            data = r.json().get("data", {})
            total = data.get("quote", {}).get(
                "USD", {}).get("total_market_cap")
            btc_dominance = data.get("btc_dominance")  # %
            eth_dominance = data.get("eth_dominance")  # %
            if total is None or btc_dominance is None or eth_dominance is None:
                return None
            btc_mcap = total * (btc_dominance / 100.0)
            eth_mcap = total * (eth_dominance / 100.0)
            total3 = total - btc_mcap - eth_mcap
            return total3
        except Exception as e:
            log.warning("cmc_total3 failed: %s", e)
            return None

# ---------- Technicals ----------


def rsi(values, length=14):
    vals = [v for v in values if v is not None]
    if len(vals) < length + 1:
        return None
    gains = []
    losses = []
    for i in range(1, len(vals)):
        diff = vals[i] - vals[i-1]
        gains.append(max(0.0, diff))
        losses.append(max(0.0, -diff))
    # Wilder's smoothing
    avg_gain = sum(gains[:length]) / length
    avg_loss = sum(losses[:length]) / length
    for i in range(length, len(gains)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def ma(series, length):
    if series is None or len(series) < length:
        return None
    return sum(series[-length:]) / float(length)


def stoch_rsi(values, length=14, k_smooth=3, d_smooth=3):
    """Return (K,D) last values. K/D in [0..1]."""
    # Build RSI series
    vals = [v for v in values if v is not None]
    if len(vals) < length + 2:
        return (None, None)
    # Make rolling RSI values (simple approach: RSI over the trailing length each step)
    rsis = []
    for i in range(length, len(vals)):
        r = rsi(vals[:i+1], length)
        if r is not None:
            rsis.append(r)
    if len(rsis) < length:
        return (None, None)

    def roll_min_max(arr, w):
        w = min(w, len(arr))
        sub = arr[-w:]
        return min(sub), max(sub)
    lo, hi = roll_min_max(rsis, length)
    if hi - lo == 0:
        raw_k = 0.5
    else:
        raw_k = (rsis[-1] - lo) / (hi - lo)
    # smooth K and D

    def sma_last(arr, w):
        w = min(w, len(arr))
        return sum(arr[-w:]) / w
    # build small buffers for smoothing
    k_buf = []
    for i in range(length, len(rsis)+1):
        lo_i, hi_i = roll_min_max(rsis[:i], length)
        if hi_i - lo_i == 0:
            k_i = 0.5
        else:
            k_i = (rsis[i-1] - lo_i) / (hi_i - lo_i)
        k_buf.append(k_i)
    k_sm = sma_last(k_buf, k_smooth)
    # D smooth over K
    d_buf = []
    for i in range(len(k_buf)):
        d_buf.append(sma_last(k_buf[:i+1], k_smooth))
    d_sm = sma_last(d_buf, d_smooth)
    return (k_sm, d_sm)


def resample_2w_from_weekly(closes_weekly):
    """Take weekly closes and keep every 2nd as a 2W close series (oldest -> newest)."""
    if not closes_weekly or len(closes_weekly) < 4:
        return []
    return [closes_weekly[i] for i in range(0, len(closes_weekly), 2)]


def fib_1272_proximity(current_price, swing_low, swing_high):
    """Distance to 1.272 extension above swing_high (uptrend context)."""
    if None in (current_price, swing_low, swing_high):
        return None
    if swing_high <= swing_low:
        return None
    ext = swing_high + 1.272 * (swing_high - swing_low)
    if ext <= 0:
        return None
    dist = abs(current_price - ext) / ext * 100.0
    # clamp insane distances
    return min(dist, 100.0), ext

# ---------- Pi Cycle (OKX 1D BTC) ----------


def sma(values, length):
    if values is None or len(values) < length:
        return None
    return sum(values[-length:]) / float(length)


def compute_pi_cycle_from_closes(closes_1d):
    out = {"ma111": None, "ma350x2": None, "ratio": None,
           "proximity_pct": None, "trend": "n/a"}
    if not closes_1d or len(closes_1d) < 350:
        return out
    m111 = sma(closes_1d, 111)
    m350 = sma(closes_1d, 350)
    if m111 is None or m350 is None:
        return out
    m350x2 = 2.0 * m350
    if m350x2 <= 0:
        return out
    ratio = m111 / m350x2
    prox = abs(ratio - 1.0) * 100.0
    out.update({"ma111": m111, "ma350x2": m350x2, "ratio": ratio,
               "proximity_pct": prox, "trend": ("above" if ratio >= 1.0 else "below")})
    return out


def pi_color(ratio):
    if ratio is None:
        return "🟡"
    if ratio >= 1.00:
        return "🔴"
    if ratio >= 0.99:
        return "🟡"
    return "🟢"


def pi_subscore(pi):
    ratio = pi.get("ratio") if isinstance(pi, dict) else None
    if ratio is None:
        return 0
    if ratio >= 1.02:
        return 100
    if ratio >= 1.00:
        return int(75 + (ratio - 1.00) / 0.02 * 25)
    if ratio >= 0.99:
        return int(40 + (ratio - 0.99) / 0.01 * 35)
    if ratio >= 0.98:
        return int((ratio - 0.98) / 0.01 * 20)
    return 0

# ---------- Metrics gather ----------


async def gather_metrics(dc: DataClient):
    out = {}
    # Coingecko global
    g = await dc.coingecko_global()
    total_mcap = (g.get("total_market_cap") or {}).get("usd")
    mc_pct = g.get("market_cap_percentage") or {}
    btc_dom = mc_pct.get("btc")
    eth_dom = mc_pct.get("eth")
    btc_mcap = None
    eth_mcap = None
    altcap_ratio = None
    if total_mcap and btc_dom is not None:
        btc_mcap = total_mcap * (btc_dom/100.0)
        if btc_mcap > 0:
            altcap_ratio = (total_mcap - btc_mcap) / btc_mcap

    prices = await dc.cg_simple_price(("bitcoin", "ethereum"), "usd")
    eth_btc = None
    try:
        if "bitcoin" in prices and "ethereum" in prices:
            btc_usd = float(prices["bitcoin"]["usd"])
            eth_usd = float(prices["ethereum"]["usd"])
            if btc_usd > 0:
                eth_btc = eth_usd / btc_usd
    except Exception:
        pass

    # Funding basket (OKX)
    funding_vals = []
    funding_detail = []
    for inst in CFG["funding_symbols"]:
        fr = await dc.okx_funding_rate(inst)
        if fr is not None:
            funding_vals.append(fr)
        funding_detail.append((inst, fr))
    funding_max = None if not funding_vals else max(
        abs(v) for v in funding_vals)
    funding_median = None
    if funding_vals:
        # median of absolute values
        funding_median = statistics.median([abs(v) for v in funding_vals])

    # OI BTC/ETH
    oi_btc = await dc.okx_open_interest_usd("BTC-USDT-SWAP")
    oi_eth = await dc.okx_open_interest_usd("ETH-USDT-SWAP")

    # Google Trends 7d avg
    trends_7 = dc.google_trends_score(
        ("crypto", "bitcoin", "ethereum"), days=7)

    # Fear & Greed now + history
    fg_now = None
    fg14 = None
    fg30 = None
    greed_run = 0
    greed_pct30 = 0
    try:
        now = await dc.fear_greed(limit=1)
        data_now = (now.get("data") or [])
        if data_now:
            fg_now = safe_float(data_now[0].get("value"))
        hist = await dc.fear_greed(limit=60)
        hist_data = hist.get("data") or []
        vals = [safe_float(d.get("value"))
                for d in hist_data if safe_float(d.get("value")) is not None]
        vals = list(reversed(vals))  # oldest->newest
        if vals:
            # Greed persistence: count consecutive >=70 from end
            for v in reversed(vals):
                if v is not None and v >= 70:
                    greed_run += 1
                else:
                    break
            last30 = vals[-30:] if len(vals) >= 30 else vals
            if last30:
                greed_pct30 = int(
                    round(100.0 * sum(1 for v in last30 if v >= 70) / len(last30)))
            last14 = vals[-14:] if len(vals) >= 14 else vals
            if last14:
                fg14 = sum(last14) / len(last14)
            if last30:
                fg30 = sum(last30) / len(last30)
    except Exception as e:
        log.warning("F&G processing failed: %s", e)

    # Pi Cycle using OKX 1D BTC
    btc_1d = await dc.okx_btc_daily_closes(limit=500)
    pi = compute_pi_cycle_from_closes(btc_1d) if btc_1d else {
        "ratio": None, "proximity_pct": None, "trend": "n/a"}

    # Weekly/2W momentum (BTC, ETH/BTC, ALT basket)
    btc_w = await dc.okx_candles("BTC-USDT", bar="1W", limit=400)
    ethbtc_w = await dc.okx_candles("ETH-BTC", bar="1W", limit=400)

    # ALT basket weekly closes: average normalized index
    alt_closes_map = {}
    for sym in CFG["alt_basket"]:
        alt_closes_map[sym] = await dc.okx_candles(sym, bar="1W", limit=400)
    # build equal-weight index
    eqw_series = []
    # normalize each coin to 1 at start of available common window
    min_len = min((len(v) for v in alt_closes_map.values() if v), default=0)
    if min_len >= 50:
        idx_matrix = []
        for v in alt_closes_map.values():
            v2 = v[-min_len:]
            base = v2[0]
            if base and base > 0:
                idx = [x/base for x in v2]
                idx_matrix.append(idx)
        if idx_matrix:
            # equal weight mean per step
            for i in range(min_len):
                eqw_series.append(sum(row[i]
                                  for row in idx_matrix)/len(idx_matrix))

    # 2W resample
    btc_2w = resample_2w_from_weekly(btc_w)
    ethbtc_2w = resample_2w_from_weekly(ethbtc_w)
    alt_2w = resample_2w_from_weekly(eqw_series)

    # RSI and MA on RSI (use 14, MA len 10)
    rsi_len = 14
    rsi_ma_len = 10
    rsi_btc_2w = rsi(btc_2w, rsi_len)
    rsi_btc_2w_ma = None
    if len(btc_2w) >= (rsi_len + rsi_ma_len + 1):
        # build RSI series for last rsi_ma_len
        rsis = []
        for i in range(rsi_len, len(btc_2w)+1):
            rsis.append(rsi(btc_2w[:i], rsi_len))
        rsis = [x for x in rsis if x is not None]
        if len(rsis) >= rsi_ma_len:
            rsi_btc_2w_ma = sum(rsis[-rsi_ma_len:]) / rsi_ma_len

    rsi_ethbtc_2w = rsi(ethbtc_2w, rsi_len)
    rsi_ethbtc_2w_ma = None
    if len(ethbtc_2w) >= (rsi_len + rsi_ma_len + 1):
        rsis = []
        for i in range(rsi_len, len(ethbtc_2w)+1):
            rsis.append(rsi(ethbtc_2w[:i], rsi_len))
        rsis = [x for x in rsis if x is not None]
        if len(rsis) >= rsi_ma_len:
            rsi_ethbtc_2w_ma = sum(rsis[-rsi_ma_len:]) / rsi_ma_len

    rsi_alt_2w = rsi(alt_2w, rsi_len) if alt_2w else None
    rsi_alt_2w_ma = None
    if alt_2w and len(alt_2w) >= (rsi_len + rsi_ma_len + 1):
        rsis = []
        for i in range(rsi_len, len(alt_2w)+1):
            rsis.append(rsi(alt_2w[:i], rsi_len))
        rsis = [x for x in rsis if x is not None]
        if len(rsis) >= rsi_ma_len:
            rsi_alt_2w_ma = sum(rsis[-rsi_ma_len:]) / rsi_ma_len

    # Stoch RSI (2W)
    k_btc_2w, d_btc_2w = stoch_rsi(btc_2w, length=14, k_smooth=3, d_smooth=3)
    k_alt_2w, d_alt_2w = stoch_rsi(
        alt_2w, length=14, k_smooth=3, d_smooth=3) if alt_2w else (None, None)

    # Fibonacci proximity on 1W
    fib_btc = (None, None)
    fib_alt = (None, None)
    try:
        if btc_w and len(btc_w) >= 60:
            cur = btc_w[-1]
            look = btc_w[-60:]
            swing_low = min(look)
            swing_high = max(look)
            fb = fib_1272_proximity(cur, swing_low, swing_high)
            if fb:
                fib_btc = fb
    except Exception:
        pass
    try:
        if eqw_series and len(eqw_series) >= 60:
            cur = eqw_series[-1]
            look = eqw_series[-60:]
            swing_low = min(look)
            swing_high = max(look)
            fa = fib_1272_proximity(cur, swing_low, swing_high)
            if fa:
                fib_alt = fa
    except Exception:
        pass

    # Optional TOTAL3 via CMC
    total3 = await dc.cmc_total3()

    out.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "risk_profile": CFG.get("risk_profile", "moderate"),
        "btc_dominance": btc_dom,
        "eth_btc": eth_btc,
        "altcap_btc_ratio": altcap_ratio,
        "funding_max": None if funding_max is None else (funding_max * 100.0),
        "funding_median": None if funding_median is None else (funding_median * 100.0),
        "funding_detail": funding_detail,
        "oi_btc_usd": oi_btc,
        "oi_eth_usd": oi_eth,
        "trends_7": trends_7,
        "fng_now": fg_now,
        "fng_ma14": fg14,
        "fng_ma30": fg30,
        "fng_greed_run": greed_run,
        "fng_greed_pct30": greed_pct30,
        "pi_cycle": pi,
        "rsi_btc_2w": rsi_btc_2w,
        "rsi_btc_2w_ma": rsi_btc_2w_ma,
        "rsi_ethbtc_2w": rsi_ethbtc_2w,
        "rsi_ethbtc_2w_ma": rsi_ethbtc_2w_ma,
        "rsi_alt_2w": rsi_alt_2w,
        "rsi_alt_2w_ma": rsi_alt_2w_ma,
        "stoch_btc_2w_k": k_btc_2w, "stoch_btc_2w_d": d_btc_2w,
        "stoch_alt_2w_k": k_alt_2w, "stoch_alt_2w_d": d_alt_2w,
        "fib_btc": fib_btc,  # (distance %, level)
        "fib_alt": fib_alt,
        "total3": total3
    })
    return out

# ---------- Composite scoring ----------


def scale_01(x, lo, hi):
    if x is None:
        return 0.0
    if hi == lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


def subscores_from_metrics(m):
    subs = {}

    # Alt structure (higher ratios => closer to alt top risk)
    subs["altcap_vs_btc"] = int(
        scale_01(m.get("altcap_btc_ratio"), 1.2, 1.8) * 100)
    subs["eth_btc"] = int(scale_01(m.get("eth_btc"), 0.07, 0.09) * 100)

    # Funding (abs %)
    subs["funding"] = int(scale_01(m.get("funding_max"), 0.08, 0.10) * 100)

    # OI caps
    subs["OI_BTC"] = int(
        scale_01((m.get("oi_btc_usd") or 0)/1e9, 16, 20) * 100)  # in $B
    subs["OI_ETH"] = int(
        scale_01((m.get("oi_eth_usd") or 0)/1e9, 6.4, 8.0) * 100)

    # Trends (Google)
    subs["trends"] = int(scale_01(m.get("trends_7"), 60, 75) * 100)

    # Fear & Greed family (note: 56..70 scaled to 0..100)
    def fng_score(v, lo, hi):
        return int(scale_01(v, lo, hi) * 100)
    subs["fng_now"] = fng_score(m.get("fng_now"), 56, 70)
    subs["fng_ma14"] = fng_score(m.get("fng_ma14"), 56, 70)
    subs["fng_ma30"] = fng_score(m.get("fng_ma30"), 52, 65)
    # persistence (either high run of days OR high % last 30)
    run = m.get("fng_greed_run") or 0
    pct30 = m.get("fng_greed_pct30") or 0
    run_s = scale_01(run, 8, 10)
    pct_s = scale_01(pct30, 48, 60)
    subs["fng_persist"] = int(100 * max(run_s, pct_s))

    # Pi Cycle
    subs["pi"] = pi_subscore(m.get("pi_cycle", {}))

    # Momentum bits (2W BTC/ALT + 1W fib)
    subs["RSI_BTC_2W"] = int(scale_01(m.get("rsi_btc_2w"), 60, 70) * 100)
    subs["RSI_ETHBTC_2W"] = int(scale_01(m.get("rsi_ethbtc_2w"), 55, 65) * 100)
    # Stoch: risk when overbought (>=0.8) and bearish cross (K<D)
    k_b, d_b = m.get("stoch_btc_2w_k"), m.get("stoch_btc_2w_d")
    if k_b is None or d_b is None:
        subs["Stoch_BTC_2W"] = 0
    else:
        ob = 1 if (k_b >= 0.8 or d_b >= 0.8) else 0
        cross = 1 if (k_b < d_b and (k_b >= 0.8 or d_b >= 0.8)) else 0
        subs["Stoch_BTC_2W"] = int(100 * (0.5*ob + 0.5*cross))

    # ALT RSI/Stoch
    subs["RSI_ALT_2W"] = int(scale_01(m.get("rsi_alt_2w"), 65, 75) * 100)
    k_a, d_a = m.get("stoch_alt_2w_k"), m.get("stoch_alt_2w_d")
    if k_a is None or d_a is None:
        subs["Stoch_ALT_2W"] = 0
    else:
        ob = 1 if (k_a >= 0.8 or d_a >= 0.8) else 0
        cross = 1 if (k_a < d_a and (k_a >= 0.8 or d_a >= 0.8)) else 0
        subs["Stoch_ALT_2W"] = int(100 * (0.5*ob + 0.5*cross))

    # Fib proximity (distance <= thresholds => high risk)
    fib_btc = m.get("fib_btc") or (None, None)
    fib_alt = m.get("fib_alt") or (None, None)
    dist_btc = fib_btc[0]
    dist_alt = fib_alt[0]
    subs["Fib_BTC"] = int(scale_01(
        3.0 - clamp(dist_btc if dist_btc is not None else 999, 0, 3.0), 0.0, 3.0) * 100)
    subs["Fib_ALT"] = int(scale_01(
        3.0 - clamp(dist_alt if dist_alt is not None else 999, 0, 3.0), 0.0, 3.0) * 100)

    return subs


def composite_certainty(m, profile="moderate"):
    """Return (score0..100, top5 list[(name,score,contrib)], flags[list[str]])."""
    subs = subscores_from_metrics(m)

    # profile weights
    base_w = {
        "altcap_vs_btc": 0.10, "eth_btc": 0.10,
        "funding": 0.08, "OI_BTC": 0.10, "OI_ETH": 0.05,
        "trends": 0.08,
        "fng_now": 0.07, "fng_ma14": 0.07, "fng_ma30": 0.08, "fng_persist": 0.07,
        "pi": 0.10,
        "RSI_BTC_2W": 0.05, "RSI_ETHBTC_2W": 0.05,
        "Stoch_BTC_2W": 0.05, "RSI_ALT_2W": 0.05, "Stoch_ALT_2W": 0.05,
        "Fib_BTC": 0.05, "Fib_ALT": 0.05
    }
    if profile == "conservative":
        # slightly more weight on sentiment/derivs
        base_w["fng_persist"] += 0.03
        base_w["funding"] += 0.02
        base_w["OI_BTC"] += 0.02
    elif profile == "aggressive":
        # more on momentum
        base_w["RSI_ALT_2W"] += 0.03
        base_w["Stoch_ALT_2W"] += 0.03
        base_w["RSI_ETHBTC_2W"] += 0.02

    # normalize weights to sum 1
    total_w = sum(base_w.values())
    for k in base_w:
        base_w[k] = base_w[k] / total_w

    # compute score
    parts = []
    score = 0.0
    for k, w in base_w.items():
        s = subs.get(k, 0)
        contrib = w * s
        score += contrib
        parts.append((k, s, contrib))
    parts.sort(key=lambda t: t[2], reverse=True)
    top5 = parts[:5]

    # threshold colors
    color = "🟢"
    if score >= 70:
        color = "🔴"
    elif score >= 40:
        color = "🟡"

    # flags
    flags = []
    if (m.get("fng_now") or 0) >= 70:
        flags.append("Greed is elevated (Fear & Greed Index)")
    if (m.get("fng_ma30") or 0) >= 65:
        flags.append("Fear & Greed 30-day avg in Greed")
    if (m.get("fng_greed_run") or 0) >= 10 or (m.get("fng_greed_pct30") or 0) >= 60:
        flags.append("Greed persistence is high")

    return (int(round(score)), top5, color, subs)

# ---------- Formatting ----------


def fmt_line(label, value_str, color_icon, thresholds_str):
    return f"• {label}: {color_icon} {value_str}  ({thresholds_str})"


def format_snapshot(m):
    lines = []
    ts = m.get("timestamp", datetime.now(timezone.utc).isoformat())
    profile = m.get("risk_profile", "moderate")
    lines.append(f"📊 Crypto Market Snapshot — {ts} UTC")
    lines.append(f"Profile: {profile}")
    lines.append("")  # spacer

    # Market Structure
    lines.append("Market Structure")
    # BTC dominance
    dom = m.get("btc_dominance")
    icon = green_yellow_red(dom, 48.0, 60.0)
    dom_str = f"{dom:.2f}%" if dom is not None else "n/a"
    lines.append(fmt_line("Bitcoin market share of total crypto",
                 dom_str, icon, "warn ≥ 48.00%, flag ≥ 60.00%"))
    # ETH/BTC
    ebr = m.get("eth_btc")
    icon = green_yellow_red(ebr, 0.072, 0.090)
    ebr_str = f"{ebr:.5f}" if ebr is not None else "n/a"
    lines.append(fmt_line("Ether price relative to Bitcoin (ETH/BTC)",
                 ebr_str, icon, "warn ≥ 0.07200, flag ≥ 0.09000"))
    # Altcap/BTC
    ar = m.get("altcap_btc_ratio")
    icon = green_yellow_red(ar, 1.44, 1.80)
    ar_str = f"{ar:.2f}" if ar is not None else "n/a"
    lines.append(fmt_line("Altcoin market cap / Bitcoin market cap",
                 ar_str, icon, "warn ≥ 1.44, flag ≥ 1.80"))

    lines.append("")  # spacer

    # Derivatives
    lines.append("Derivatives")
    fmax = m.get("funding_max")
    fmed = m.get("funding_median")
    f_icon_max = green_yellow_red(fmax, 0.08, 0.10)
    f_icon_med = green_yellow_red(fmed, 0.08, 0.10)
    fmax_s = f"{fmax:.3f}%" if fmax is not None else "n/a"
    fmed_s = f"{fmed:.3f}%" if fmed is not None else "n/a"
    lines.append(
        f"• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — max: {f_icon_max} {fmax_s} | median: {f_icon_med} {fmed_s}  (warn ≥ 0.080%, flag ≥ 0.100%)")
    # top-3 funding
    detail = [(sym, val) for (sym, val) in (
        m.get("funding_detail") or []) if val is not None]
    detail_pct = sorted([(sym, abs(val)*100.0)
                        for sym, val in detail], key=lambda x: x[1], reverse=True)[:3]
    if detail_pct:
        parts = [
            f"{sym.replace('-USDT-SWAP','')} {v:.3f}%" for sym, v in detail_pct]
        lines.append("  Top-3 funding extremes: " + ", ".join(parts))
    # OI
    oi_b = m.get("oi_btc_usd")
    oi_e = m.get("oi_eth_usd")
    icon = green_yellow_red((oi_b or 0)/1e9, 16, 20)
    lines.append(fmt_line("Bitcoin open interest (USD)", usd(
        oi_b), icon, "warn ≥ $16,000,000,000, flag ≥ $20,000,000,000"))
    icon = green_yellow_red((oi_e or 0)/1e9, 6.4, 8.0)
    lines.append(fmt_line("Ether open interest (USD)", usd(oi_e),
                 icon, "warn ≥ $6,400,000,000, flag ≥ $8,000,000,000"))

    lines.append("")

    # Sentiment
    lines.append("Sentiment")
    tr = m.get("trends_7")
    icon = green_yellow_red(tr, 60.0, 75.0)
    lines.append(fmt_line("Google Trends avg (7d; crypto/bitcoin/ethereum)",
                 (f"{tr:.1f}" if tr is not None else "n/a"), icon, "warn ≥ 60.0, flag ≥ 75.0"))
    fng = m.get("fng_now")
    icon = green_yellow_red(fng, 56, 70)
    fng_str = f"{int(round(fng))} (Greed)" if fng is not None else "n/a"
    lines.append(fmt_line("Fear & Greed Index (overall crypto)",
                 fng_str, icon, "warn ≥ 56, flag ≥ 70"))
    fng14 = m.get("fng_ma14")
    icon = green_yellow_red(fng14, 56, 70)
    lines.append(fmt_line("Fear & Greed 14-day average",
                 (f"{fng14:.1f}" if fng14 is not None else "n/a"), icon, "warn ≥ 56, flag ≥ 70"))
    fng30 = m.get("fng_ma30")
    icon = green_yellow_red(fng30, 52, 65)
    lines.append(fmt_line("Fear & Greed 30-day average",
                 (f"{fng30:.1f}" if fng30 is not None else "n/a"), icon, "warn ≥ 52, flag ≥ 65"))
    run = m.get("fng_greed_run") or 0
    pct30 = m.get("fng_greed_pct30") or 0
    icon = "🔴" if run >= 10 or pct30 >= 60 else (
        "🟡" if run >= 8 or pct30 >= 48 else "🟢")
    lines.append(
        f"• Greed persistence: {icon} {run} days in a row | {pct30}% of last 30 days ≥ 70  (warn: days ≥ 8 or pct ≥ 48%; flag: days ≥ 10 or pct ≥ 60%)")

    lines.append("")

    # Cycle & On-Chain
    lines.append("Cycle & On-Chain")
    pi = m.get("pi_cycle", {})
    ratio = pi.get("ratio")
    prox = pi.get("proximity_pct")
    trend = pi.get("trend", "n/a")
    icon = pi_color(ratio)
    if ratio is None or prox is None:
        lines.append("• Pi Cycle Top proximity: 🟡 n/a")
    else:
        rel = "above" if trend == "above" else "below"
        lines.append(
            f"• Pi Cycle Top proximity: {icon} {prox:.2f}% {rel} (111D vs 2×350D)")

    lines.append("")

    # Momentum & Extensions
    lines.append("Momentum (2W) & Extensions (1W)")
    # BTC RSI
    rb = m.get("rsi_btc_2w")
    rb_ma = m.get("rsi_btc_2w_ma")
    icon = green_yellow_red(rb, 60, 70)
    rb_str = f"{rb:.1f}" if rb is not None else "n/a"
    rbma_str = f"{rb_ma:.1f}" if rb_ma is not None else "n/a"
    lines.append(
        f"• BTC RSI (2W): {icon} {rb_str} (MA {rbma_str}) (warn ≥ 60.0, flag ≥ 70.0)")
    # ETH/BTC RSI
    re = m.get("rsi_ethbtc_2w")
    re_ma = m.get("rsi_ethbtc_2w_ma")
    icon = green_yellow_red(re, 55, 65)
    re_str = f"{re:.1f}" if re is not None else "n/a"
    rema_str = f"{re_ma:.1f}" if re_ma is not None else "n/a"
    lines.append(
        f"• ETH/BTC RSI (2W): {icon} {re_str} (MA {rema_str}) (warn ≥ 55.0, flag ≥ 65.0)")
    # ALT RSI
    ra = m.get("rsi_alt_2w")
    ra_ma = m.get("rsi_alt_2w_ma")
    icon = green_yellow_red(ra, 65, 75)
    ra_str = f"{ra:.1f}" if ra is not None else "n/a"
    rama_str = f"{ra_ma:.1f}" if ra_ma is not None else "n/a"
    lines.append(
        f"• ALT basket (equal-weight) RSI (2W): {icon} {ra_str} (MA {rama_str}) (warn ≥ 65.0, flag ≥ 75.0)")
    # Stoch RSI BTC/ALT
    kb, db = m.get("stoch_btc_2w_k"), m.get("stoch_btc_2w_d")
    icon = "🔴" if (kb is not None and db is not None and (kb < db) and (kb >= 0.8 or db >= 0.8)) else (
        "🟡" if (kb is not None and db is not None and (kb >= 0.8 or db >= 0.8)) else "🟢")
    kb_s = f"{kb:.2f}" if kb is not None else "n/a"
    db_s = f"{db:.2f}" if db is not None else "n/a"
    lines.append(
        f"• BTC Stoch RSI (2W) K/D: {icon} {kb_s}/{db_s} (overbought ≥ 0.80; red = bearish cross from OB)")
    ka, da = m.get("stoch_alt_2w_k"), m.get("stoch_alt_2w_d")
    icon = "🔴" if (ka is not None and da is not None and (ka < da) and (ka >= 0.8 or da >= 0.8)) else (
        "🟡" if (ka is not None and da is not None and (ka >= 0.8 or da >= 0.8)) else "🟢")
    ka_s = f"{ka:.2f}" if ka is not None else "n/a"
    da_s = f"{da:.2f}" if da is not None else "n/a"
    lines.append(
        f"• ALT basket Stoch RSI (2W) K/D: {icon} {ka_s}/{da_s} (overbought ≥ 0.80; red = bearish cross from OB)")
    # Fib prox
    fb = m.get("fib_btc") or (None, None)
    dist_b, lvl_b = fb
    fa = m.get("fib_alt") or (None, None)
    dist_a, lvl_a = fa
    icon = green_yellow_red(
        dist_b, 3.0, 1.5, reverse=True) if dist_b is not None else "🟡"
    if dist_b is None or lvl_b is None:
        lines.append("• BTC Fibonacci extension proximity: 🟡 n/a")
    else:
        lines.append(
            f"• BTC Fibonacci extension proximity: {icon} 1.272 @ {dist_b:.2f}% away")
    icon = green_yellow_red(
        dist_a, 3.0, 1.5, reverse=True) if dist_a is not None else "🟡"
    if dist_a is None or lvl_a is None:
        lines.append("• ALT basket Fibonacci proximity: 🟡 n/a")
    else:
        lines.append(
            f"• ALT basket Fibonacci proximity: {icon} 1.272 @ {dist_a:.2f}% away")

    lines.append("")

    # Composite
    score, top5, score_icon, subs = composite_certainty(m, profile=profile)
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {score_icon} {score}/100 (yellow ≥ 40, red ≥ 70)")
    if top5:
        lines.append("• Top drivers:")
        for name, s, contrib in top5:
            # label friendly names
            label_map = {
                "altcap_vs_btc": "Altcap/BTC",
                "eth_btc": "ETH/BTC",
                "funding": "Funding",
                "OI_BTC": "BTC OI",
                "OI_ETH": "ETH OI",
                "trends": "Google Trends",
                "fng_now": "F&G (today, scaled)",
                "fng_ma14": "F&G 14-d",
                "fng_ma30": "F&G 30-d",
                "fng_persist": "F&G persistence",
                "pi": "Pi Cycle",
                "RSI_BTC_2W": "BTC RSI (2W)",
                "RSI_ETHBTC_2W": "ETH/BTC RSI (2W)",
                "Stoch_BTC_2W": "BTC Stoch RSI (2W)",
                "RSI_ALT_2W": "ALT RSI (2W)",
                "Stoch_ALT_2W": "ALT Stoch RSI (2W)",
                "Fib_BTC": "BTC Fib 1.272",
                "Fib_ALT": "ALT Fib 1.272",
            }
            label = label_map.get(name, name)
            # color each driver by magnitude
            d_icon = "🔴" if s >= 70 else ("🟡" if s >= 40 else "🟢")
            lines.append(f"• {label}: {d_icon} {s}/100")

    # Flags
    fl_score, fl_top5, fl_color, _ = composite_certainty(
        m, profile=profile)  # recompute for flags list already within
    flags = []
    if (m.get("fng_now") or 0) >= 70:
        flags.append("Greed is elevated (Fear & Greed Index)")
    if (m.get("fng_ma30") or 0) >= 65:
        flags.append("Fear & Greed 30-day avg in Greed")
    if (m.get("fng_greed_run") or 0) >= 10 or (m.get("fng_greed_pct30") or 0) >= 60:
        flags.append("Greed persistence is high")
    if flags:
        lines.append("")
        lines.append(f"⚠️ Triggered flags ({len(flags)}): " + ", ".join(flags))

    return "\n".join(lines)

# ---------- Telegram Handlers ----------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Hi! I track cycle-top risk for alts using market structure, derivatives, sentiment, "
        "Pi Cycle, and 2-week momentum.\n\n"
        "Commands:\n"
        "• /status – current assessment\n"
        "You’ll also receive a daily summary automatically."
    )
    await update.message.reply_text(txt)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        async with DataClient() as_dummy:  # type: ignore
            pass
    except Exception:
        pass
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        text = format_snapshot(m)
        await update.message.reply_text(text)
    except Exception as e:
        log.exception("status failed")
        await update.message.reply_text(f"⚠️ Could not fetch metrics right now: {e}")
    finally:
        await dc.close()

# ---------- Push jobs ----------


async def push_summary(app: Application):
    chat_id = CFG["telegram"].get("admin_chat_id") or ""
    if not chat_id:
        return
    dc = DataClient()
    try:
        m = await gather_metrics(dc)
        text = format_snapshot(m)
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        log.exception("push_summary failed: %s", e)
    finally:
        await dc.close()


async def push_alerts(app: Application):
    # Keep it simple: reuse same snapshot for now (you can tighten thresholds here)
    await push_summary(app)

# ---------- Health server ----------


async def handle_root(request):
    return web.Response(text="Crypto Cycle Bot is running.")


async def handle_health(request):
    return web.json_response({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


async def start_health_server():
    app = web.Application()
    app.router.add_get("/", handle_root)
    app.router.add_get("/health", handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    log.info("Health server listening on :8080")

# ---------- Main ----------


async def main():
    token = CFG["telegram"]["token"]
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN not set")

    await start_health_server()

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))

    # Schedulers
    scheduler = AsyncIOScheduler(timezone="UTC")
    # daily summary at HH:MM
    hh, mm = CFG.get("daily_push_utc_hhmm", "14:00").split(":")
    scheduler.add_job(lambda: asyncio.create_task(push_summary(
        app)), "cron", hour=int(hh), minute=int(mm), id="push_summary")
    # intraday alerts
    every = int(CFG.get("alerts_every_minutes", 15))
    scheduler.add_job(lambda: asyncio.create_task(push_alerts(
        app)), "cron", minute=f"*/{every}", id="push_alerts")
    scheduler.start()
    log.info("Bot running. Press Ctrl+C to exit.")

    await app.delete_webhook(drop_pending_updates=True)
    await app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
