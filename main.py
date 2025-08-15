import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aiohttp import web
from pytrends.request import TrendReq
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram import Update

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
log = logging.getLogger("crypto-cycle-bot")

# ----------------------------
# Config / Env
# ----------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # required
SUMMARY_UTC_HOUR = int(os.getenv("SUMMARY_UTC_HOUR", "15"))
ALERT_CRON_MINUTES = os.getenv("ALERT_CRON_MINUTES", "*/15")
USER_AGENT = "crypto-cycle-telegram-bot/1.0 (+bot)"

# Risk profiles (weights tweak multipliers)
RISK_PROFILES = {
    "conservative": 0.8,
    "moderate": 1.0,
    "aggressive": 1.2,
}

# Markets & baskets
FUNDING_BASKET = [
    ("BTC-USDT-SWAP", "BTCUSDT"),
    ("ETH-USDT-SWAP", "ETHUSDT"),
    ("SOL-USDT-SWAP", "SOLUSDT"),
    ("XRP-USDT-SWAP", "XRPUSDT"),
    ("DOGE-USDT-SWAP", "DOGEUSDT"),
]

ALT_BASKET = [
    "SOL-USDT", "XRP-USDT", "DOGE-USDT", "ADA-USDT",
    "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"
]

# Thresholds
THRESH = {
    "btc_dominance_warn": 0.48, "btc_dominance_flag": 0.60,
    "eth_btc_warn": 0.072, "eth_btc_flag": 0.090,
    "altcap_btc_warn": 1.44, "altcap_btc_flag": 1.80,
    "funding_warn": 0.0008, "funding_flag": 0.0010,  # 0.08% / 0.10%
    "oi_btc_warn": 16e9, "oi_btc_flag": 20e9,
    "oi_eth_warn": 6.4e9, "oi_eth_flag": 8e9,
    "trends_warn": 60.0, "trends_flag": 75.0,
    "fng_warn": 56, "fng_flag": 70,
    "fng14_warn": 56, "fng14_flag": 70,
    "fng30_warn": 52, "fng30_flag": 65,
    "fng_persist_days_warn": 8, "fng_persist_days_flag": 10,
    "fng_persist_pct_warn": 0.48, "fng_persist_pct_flag": 0.60,
    "rsi_btc2w_warn": 60.0, "rsi_btc2w_flag": 70.0,
    "rsi_ethbtc2w_warn": 55.0, "rsi_ethbtc2w_flag": 65.0,
    "rsi_alt2w_warn": 65.0, "rsi_alt2w_flag": 75.0,
    "fib_warn": 0.03, "fib_flag": 0.015,
}

# Runtime state (in-memory)
SUBSCRIBERS: set[int] = set()
CHAT_PROFILE: Dict[int, str] = {}  # chat_id -> profile

# ----------------------------
# Utilities
# ----------------------------


def to_human_dollar(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if v >= 1e12:
        return f"${v/1e12:.3f}T"
    if v >= 1e9:
        return f"${v/1e9:.3f}B"
    if v >= 1e6:
        return f"${v/1e6:.3f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def col(flag: str) -> str:
    return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(flag, "🟢")


def tri_flag(value: Optional[float], warn: float, flag: float, higher_is_risk: bool = True) -> str:
    if value is None:
        return "yellow"
    v = float(value)
    if higher_is_risk:
        if v >= flag:
            return "red"
        if v >= warn:
            return "yellow"
        return "green"
    else:
        if v <= flag:
            return "red"
        if v <= warn:
            return "yellow"
        return "green"


def simple_sma(arr: List[float], n: int) -> Optional[float]:
    if len(arr) < n:
        return None
    return float(np.mean(arr[-n:]))


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    deltas = np.diff(np.array(values, dtype=float))
    gains = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def stoch_rsi(values: List[float], period: int = 14, k: int = 3, d: int = 3) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < period + max(k, d) + 1:
        return (None, None)
    rsi_vals = []
    for i in range(period, len(values)):
        r = rsi(values[: i + 1], period)
        if r is not None:
            rsi_vals.append(r)
    if len(rsi_vals) < period:
        return (None, None)
    srsi = []
    for i in range(period, len(rsi_vals) + 1):
        window = rsi_vals[i - period: i]
        lo = min(window)
        hi = max(window)
        srsi.append(0.5 if hi - lo ==
                    0 else (rsi_vals[i - 1] - lo) / (hi - lo))
    if len(srsi) < k + d:
        return (None, None)
    k_vals = pd.Series(srsi).rolling(k).mean().dropna().tolist()
    if not k_vals:
        return (None, None)
    k_last = k_vals[-1]
    d_vals = pd.Series(k_vals).rolling(d).mean().dropna().tolist()
    if not d_vals:
        return (None, None)
    d_last = d_vals[-1]
    return (float(k_last), float(d_last))


def downsample_2w(closes: List[float]) -> List[float]:
    if not closes:
        return []
    return closes[1::2]  # oldest->newest weekly -> 2W


def fib_extension_proximity(weekly_closes: List[float], lookback: int = 52) -> Optional[Tuple[float, float]]:
    if len(weekly_closes) < lookback + 1:
        return None
    data = weekly_closes[-lookback:]
    lo = float(min(data))
    hi = float(max(data))
    if hi <= 0:
        return None
    rng = hi - lo
    ext1272 = hi + 1.272 * rng
    last = float(weekly_closes[-1])
    away = abs(ext1272 - last) / ext1272
    return (ext1272, away)


def pct_str(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.{decimals}f}%"


def parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None

# ----------------------------
# Data Client
# ----------------------------


class DataClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(
            25.0), headers={"User-Agent": USER_AGENT})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            aclose = getattr(self.client, "aclose", None)
            if aclose:
                await aclose()
                return
            close = getattr(self.client, "close", None)
            if close:
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()
        except Exception as e:
            log.warning("http client close error: %s", e)

    # ---- OKX helpers ----
    async def okx_funding_and_ticker(self, inst_id: str) -> Tuple[Optional[float], Optional[float]]:
        try:
            fr = await self.client.get("https://www.okx.com/api/v5/public/funding-rate", params={"instId": inst_id})
            fr.raise_for_status()
            f_data = fr.json().get("data", [])
            rate = float(f_data[0].get("fundingRate")) if f_data else None
        except Exception as e:
            log.warning("funding fetch failed for %s: %s", inst_id, e)
            rate = None
        last_price = None
        try:
            tk = await self.client.get("https://www.okx.com/api/v5/market/ticker", params={"instId": inst_id})
            tk.raise_for_status()
            t_data = tk.json().get("data", [])
            last_price = float(t_data[0].get("last")) if t_data else None
        except Exception as e:
            log.warning("ticker fetch failed for %s: %s", inst_id, e)
        return (rate, last_price)

    async def okx_open_interest_usd(self, inst_ids: List[str]) -> Optional[float]:
        total = 0.0
        got_any = False
        last_cache: Dict[str, float] = {}
        for inst_id in inst_ids:
            try:
                r = await self.client.get(
                    "https://www.okx.com/api/v5/public/open-interest",
                    params={"instType": "SWAP", "instId": inst_id}
                )
                r.raise_for_status()
                data = r.json().get("data", [])
                for row in data:
                    oi_usd_str = row.get("oiUsd")
                    if oi_usd_str not in (None, "", "0"):
                        total += float(oi_usd_str)
                        got_any = True
                        continue
                    oi_ccy = float(row.get("oiCcy", 0.0) or 0.0)
                    if oi_ccy > 0.0:
                        base = inst_id.split("-")[0]
                        spot = f"{base}-USDT"
                        if spot not in last_cache:
                            try:
                                tk = await self.client.get("https://www.okx.com/api/v5/market/ticker", params={"instId": spot})
                                tk.raise_for_status()
                                t_data = tk.json().get("data", [])
                                if t_data:
                                    last_cache[spot] = float(
                                        t_data[0].get("last"))
                            except Exception:
                                last_cache[spot] = 0.0
                        last_p = last_cache.get(spot, 0.0)
                        if last_p > 0.0:
                            total += oi_ccy * last_p
                            got_any = True
                            continue
                    oi = float(row.get("oi", 0.0) or 0.0)
                    ct_val = float(row.get("ctVal", 0.0) or 0.0)
                    if oi > 0.0 and ct_val > 0.0:
                        total += oi * ct_val
                        got_any = True
            except Exception as e:
                log.warning(
                    "open interest fetch failed for %s: %s", inst_id, e)
        return total if got_any else None

    async def okx_candles(self, inst_id: str, bar: str, limit: int = 500) -> Optional[List[Tuple[int, float]]]:
        # Returns [(ts_ms, close)] oldest->newest
        try:
            r = await self.client.get("https://www.okx.com/api/v5/market/candles",
                                      params={"instId": inst_id, "bar": bar, "limit": str(limit)})
            r.raise_for_status()
            rows = r.json().get("data", [])
            if not rows:
                return None
            # OKX returns newest first: [ts, o, h, l, c, ...]
            out = [(int(x[0]), float(x[4])) for x in rows]
            out.reverse()
            return out
        except Exception as e:
            log.warning("okx candles failed for %s %s: %s", inst_id, bar, e)
            return None

    # ---- Coinbase fallback ----
    async def coinbase_daily(self, product_id: str, limit: int = 500) -> Optional[List[Tuple[int, float]]]:
        # Coinbase returns [time, low, high, open, close, volume] newest->oldest (seconds)
        try:
            r = await self.client.get(
                f"https://api.exchange.coinbase.com/products/{product_id}/candles",
                params={"granularity": 86400, "limit": str(limit)}
            )
            r.raise_for_status()
            rows = r.json()
            if not rows:
                return None
            rows.sort(key=lambda z: z[0])  # oldest->newest
            return [(int(x[0]) * 1000, float(x[4])) for x in rows]
        except Exception as e:
            log.warning("coinbase daily candles failed for %s: %s",
                        product_id, e)
            return None

    # ---- Kraken fallback ----
    async def kraken_daily(self, pair: str, limit: int = 720) -> Optional[List[Tuple[int, float]]]:
        # Kraken OHLC: result[pair] = [[time, open, high, low, close, vwap, volume, count], ...] time in seconds
        try:
            r = await self.client.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": 1440}
            )
            r.raise_for_status()
            data = r.json().get("result", {})
            # pick the first array in result
            arr = None
            for k, v in data.items():
                if isinstance(v, list):
                    arr = v
                    break
            if not arr:
                return None
            arr = arr[-limit:]
            arr.sort(key=lambda z: z[0])
            return [(int(x[0]) * 1000, float(x[4])) for x in arr]
        except Exception as e:
            log.warning("kraken daily candles failed for %s: %s", pair, e)
            return None

    # ---- Other data (current only) ----
    async def coingecko_global(self) -> Optional[Dict[str, Any]]:
        try:
            r = await self.client.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            return r.json().get("data", {})
        except Exception as e:
            log.warning("coingecko global failed: %s", e)
            return None

    async def coingecko_prices(self, ids: List[str]) -> Dict[str, float]:
        try:
            r = await self.client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": ",".join(ids), "vs_currencies": "usd"}
            )
            r.raise_for_status()
            data = r.json()
            out: Dict[str, float] = {}
            for cid in ids:
                usd = data.get(cid, {}).get("usd")
                if usd is not None:
                    out[cid] = float(usd)
            return out
        except Exception as e:
            log.warning("coingecko simple price failed: %s", e)
            return {}

    async def fear_greed(self) -> Optional[Dict[str, Any]]:
        try:
            r1 = await self.client.get("https://api.alternative.me/fng/", params={"limit": "1"})
            r1.raise_for_status()
            now_data = r1.json().get("data", [])
            now_val = int(now_data[0].get("value")) if now_data else None

            r2 = await self.client.get("https://api.alternative.me/fng/", params={"limit": "60"})
            r2.raise_for_status()
            hist = r2.json().get("data", [])
            vals = [int(x.get("value")) for x in hist if x.get(
                "value") is not None][::-1]  # oldest->newest
            ma14 = float(np.mean(vals[-14:])) if len(vals) >= 14 else None
            ma30 = float(np.mean(vals[-30:])) if len(vals) >= 30 else None
            greedy_days = sum(
                1 for v in vals[-30:] if v >= 70) if len(vals) >= 30 else None
            greedy_pct = (
                greedy_days / 30.0) if greedy_days is not None else None
            streak = 0
            if vals:
                for v in reversed(vals):
                    if v >= 70:
                        streak += 1
                    else:
                        break
            return {
                "now": now_val,
                "ma14": ma14,
                "ma30": ma30,
                "greedy_days30": greedy_days,
                "greedy_pct30": greedy_pct,
                "greed_streak": streak,
            }
        except Exception as e:
            log.warning("fear & greed fetch failed: %s", e)
            return None

    async def google_trends_score(self) -> Optional[float]:
        def _fetch():
            try:
                pytrends = TrendReq(hl="en-US", tz=0)
                kw = ["crypto", "bitcoin", "ethereum"]
                pytrends.build_payload(kw, timeframe="now 7-d", geo="")
                df = pytrends.interest_over_time()
                if df is None or df.empty:
                    return None
                df = df[kw]
                return float(df.tail(7).mean().mean())
            except Exception as e:
                log.warning("pytrends error: %s", e)
                return None
        return await asyncio.to_thread(_fetch)

# ----------------------------
# Metric Builder
# ----------------------------


async def build_metrics(as_of: Optional[datetime] = None) -> Dict[str, Any]:
    """
    If as_of is provided (UTC), backtest mode:
      - Only price-derived metrics are computed up to that date.
      - Market-cap, derivatives, and sentiment (non-price) are omitted (None).
    Otherwise, full live snapshot.
    """
    ts_cut = int(as_of.timestamp() * 1000) if as_of else None

    async with DataClient() as dc:
        # ---------- Market structure (live only) ----------
        btc_dom = None
        altcap_btc_ratio = None
        eth_btc = None

        if as_of is None:
            cg = await dc.coingecko_global()
            if cg:
                md = cg.get("market_cap_percentage", {})
                total = cg.get("total_market_cap", {}).get("usd")
                btc_mcap = None
                if total and isinstance(total, (int, float)):
                    btc_pct = md.get("btc")
                    if btc_pct is not None:
                        btc_dom = float(btc_pct) / 100.0
                        btc_mcap = total * btc_dom
                    if btc_mcap and btc_mcap > 0:
                        altcap_btc_ratio = (
                            total - btc_mcap) / btc_mcap if total else None
            prices = await dc.coingecko_prices(["bitcoin", "ethereum"])
            btc_usd = prices.get("bitcoin")
            eth_usd = prices.get("ethereum")
            eth_btc = (eth_usd / btc_usd) if (eth_usd and btc_usd) else None
        else:
            # Backtest ETH/BTC from weekly candles up to date
            ethbtc_w = await dc.okx_candles("ETH-BTC", "1W", 400)
            if ethbtc_w:
                series = [c for (t, c) in ethbtc_w if t <= ts_cut]
                eth_btc = float(series[-1]) if series else None

        # ---------- Derivatives (live only) ----------
        funding_max = None
        funding_median = None
        oi_btc = None
        oi_eth = None
        if as_of is None:
            basket_rates: List[float] = []
            for inst_id, _alias in FUNDING_BASKET:
                rate, _lp = await dc.okx_funding_and_ticker(inst_id)
                if rate is not None:
                    basket_rates.append(rate)
            funding_max = max(basket_rates) if basket_rates else None
            funding_median = float(
                np.median(basket_rates)) if basket_rates else None

            oi_btc = await dc.okx_open_interest_usd(["BTC-USDT-SWAP", "BTC-USD-SWAP"])
            oi_eth = await dc.okx_open_interest_usd(["ETH-USDT-SWAP", "ETH-USD-SWAP"])

        # ---------- Sentiment (live only) ----------
        trends = None
        fng = None
        if as_of is None:
            trends = await dc.google_trends_score()
            fng = await dc.fear_greed()

        # ---------- Momentum weekly & 2W (price-derived; slice for as_of) ----------
        btc_w_pairs = await dc.okx_candles("BTC-USDT", "1W", 400) or []
        ethbtc_w_pairs = await dc.okx_candles("ETH-BTC", "1W", 400) or []
        alt_closes_list: List[List[Tuple[int, float]]] = []
        for sym in ALT_BASKET:
            p = await dc.okx_candles(sym, "1W", 400)
            if p:
                alt_closes_list.append(p)

        def slice_to_date(pairs: List[Tuple[int, float]]) -> List[float]:
            if not pairs:
                return []
            if ts_cut is None:
                return [c for (_t, c) in pairs]
            return [c for (t, c) in pairs if t <= ts_cut]

        btc_w = slice_to_date(btc_w_pairs)
        ethbtc_w = slice_to_date(ethbtc_w_pairs)

        alt_index: Optional[List[float]] = None
        if alt_closes_list:
            # align to min length after slicing
            sliced = [[(t, c) for (t, c) in seq if (
                ts_cut is None or t <= ts_cut)] for seq in alt_closes_list]
            sliced = [s for s in sliced if len(s) >= 5]
            if sliced:
                L = min(len(s) for s in sliced)
                aligned = [s[-L:] for s in sliced]
                normed: List[List[float]] = []
                for s in aligned:
                    base = s[0][1]
                    seq = [100.0 * (v / base) for (_t, v) in s]
                    normed.append(seq)
                alt_index = list(np.mean(np.array(normed), axis=0))

        # 2W downsample
        btc_2w = downsample_2w(btc_w) if btc_w else None
        ethbtc_2w = downsample_2w(ethbtc_w) if ethbtc_w else None
        alt_2w = downsample_2w(alt_index) if alt_index else None

        rsi_btc2w = rsi(btc_2w, 14) if btc_2w else None
        rsi_ethbtc2w = rsi(ethbtc_2w, 14) if ethbtc_2w else None
        rsi_alt2w = rsi(alt_2w, 14) if alt_2w else None

        def ma_of_series(vals: Optional[List[float]], n: int = 14) -> Optional[float]:
            if not vals or len(vals) < n:
                return None
            return float(pd.Series(vals[-n:]).mean())

        rsi_btc2w_ma = None
        rsi_alt2w_ma = None
        if btc_2w and len(btc_2w) >= 30:
            seq = [rsi(btc_2w[:i], 14) or 50.0 for i in range(
                15, len(btc_2w) + 1)]
            rsi_btc2w_ma = ma_of_series(seq, 14)
        if alt_2w and len(alt_2w) >= 30:
            seq = [rsi(alt_2w[:i], 14) or 50.0 for i in range(
                15, len(alt_2w) + 1)]
            rsi_alt2w_ma = ma_of_series(seq, 14)

        # Stoch RSI on 2W
        k_btc2w, d_btc2w = (stoch_rsi(btc_2w, 14, 3, 3)
                            if btc_2w else (None, None))
        k_alt2w, d_alt2w = (stoch_rsi(alt_2w, 14, 3, 3)
                            if alt_2w else (None, None))

        # Fibonacci proximity (weekly, price-derived)
        btc_fib = fib_extension_proximity(btc_w) if btc_w else None
        alt_fib = fib_extension_proximity(alt_index) if alt_index else None

        # ---------- Pi Cycle Top proximity (daily, robust fallbacks, sliced) ----------
        btc_daily_pairs = await dc.okx_candles("BTC-USDT", "1D", 500)
        if not btc_daily_pairs:
            btc_daily_pairs = await dc.coinbase_daily("BTC-USD", 500)
        if not btc_daily_pairs:
            btc_daily_pairs = await dc.kraken_daily("XBTUSD", 720)

        btc_d = slice_to_date(btc_daily_pairs) if btc_daily_pairs else []
        pi_prox = None
        if btc_d and len(btc_d) >= 350:  # need >=350 for SMA350 (we double to emulate 700)
            ma111 = simple_sma(btc_d, 111)
            ma350 = simple_sma(btc_d, 350)
            ma700 = (ma350 * 2.0) if ma350 is not None else None
            if ma111 is not None and ma700 is not None and (ma111 + ma700) > 0:
                diff = abs(ma111 - ma700)
                denom = (ma111 + ma700) / 2.0
                pi_prox = max(0.0, min(1.0, 1.0 - (diff / denom)))

        return {
            "as_of": as_of.isoformat() if as_of else None,
            "btc_dom": btc_dom, "eth_btc": eth_btc, "altcap_btc_ratio": altcap_btc_ratio,
            "funding_max": funding_max, "funding_median": funding_median,
            "oi_btc": oi_btc, "oi_eth": oi_eth,
            "trends": trends, "fng": fng,
            "rsi_btc2w": rsi_btc2w, "rsi_ethbtc2w": rsi_ethbtc2w, "rsi_alt2w": rsi_alt2w,
            "rsi_btc2w_ma": rsi_btc2w_ma, "rsi_alt2w_ma": rsi_alt2w_ma,
            "k_btc2w": k_btc2w, "d_btc2w": d_btc2w,
            "k_alt2w": k_alt2w, "d_alt2w": d_alt2w,
            "btc_fib": btc_fib, "alt_fib": alt_fib,
            "pi_prox": pi_prox,
        }

# ----------------------------
# Composite Scoring (unchanged except uses pi_prox)
# ----------------------------


def score_component(value: Optional[float], good_low: bool, warn: float, flag: float) -> float:
    if value is None:
        return 50.0
    v = float(value)
    if good_low:
        if v >= flag:
            return 100.0
        if v <= warn:
            return 0.0
        return 100.0 * (v - warn) / (flag - warn)
    else:
        if v <= flag:
            return 100.0
        if v >= warn:
            return 0.0
        return 100.0 * (warn - v) / (warn - flag)


def alt_top_certainty(m: Dict[str, Any], profile: str) -> float:
    w = {
        "altcap_vs_btc": 0.10,
        "eth_btc": 0.08,
        "funding": 0.08,
        "oi_btc": 0.08,
        "oi_eth": 0.08,
        "trends": 0.06,
        "fng_now": 0.06,
        "fng_ma14": 0.06,
        "fng_ma30": 0.08,
        "fng_persist": 0.06,
        "pi": 0.08,
        "rsi_btc2w": 0.06,
        "rsi_ethbtc2w": 0.06,
        "rsi_alt2w": 0.06,
    }
    mult = RISK_PROFILES.get(profile, 1.0)

    s_altcap = score_component(m.get(
        "altcap_btc_ratio"), True, THRESH["altcap_btc_warn"], THRESH["altcap_btc_flag"])
    s_ethbtc = score_component(
        m.get("eth_btc"), True, THRESH["eth_btc_warn"], THRESH["eth_btc_flag"])
    s_fund = score_component(m.get("funding_max"), True,
                             THRESH["funding_warn"], THRESH["funding_flag"])
    s_oib = score_component(m.get("oi_btc"), True,
                            THRESH["oi_btc_warn"], THRESH["oi_btc_flag"])
    s_oie = score_component(m.get("oi_eth"), True,
                            THRESH["oi_eth_warn"], THRESH["oi_eth_flag"])
    s_tr = score_component(m.get("trends"), True,
                           THRESH["trends_warn"], THRESH["trends_flag"])

    fng = m.get("fng") or {}
    s_fng_now = score_component(
        fng.get("now"), True, THRESH["fng_warn"], THRESH["fng_flag"])
    s_fng14 = score_component(fng.get("ma14"), True,
                              THRESH["fng14_warn"], THRESH["fng14_flag"])
    s_fng30 = score_component(fng.get("ma30"), True,
                              THRESH["fng30_warn"], THRESH["fng30_flag"])
    persist_days = fng.get("greed_streak") or 0
    persist_pct = fng.get("greedy_pct30") or 0.0
    s_persist_days = score_component(
        persist_days, True, THRESH["fng_persist_days_warn"], THRESH["fng_persist_days_flag"])
    s_persist_pct = score_component(
        persist_pct, True, THRESH["fng_persist_pct_warn"], THRESH["fng_persist_pct_flag"])
    s_persist = 0.5 * s_persist_days + 0.5 * s_persist_pct

    pi_p = m.get("pi_prox")
    s_pi = 50.0 if pi_p is None else max(0.0, min(100.0, pi_p * 100.0))

    s_rsi_b = score_component(m.get("rsi_btc2w"), True,
                              THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"])
    s_rsi_e = score_component(m.get(
        "rsi_ethbtc2w"), True, THRESH["rsi_ethbtc2w_warn"], THRESH["rsi_ethbtc2w_flag"])
    s_rsi_a = score_component(m.get("rsi_alt2w"), True,
                              THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"])

    total = (
        w["altcap_vs_btc"] * s_altcap +
        w["eth_btc"] * s_ethbtc +
        w["funding"] * s_fund +
        w["oi_btc"] * s_oib +
        w["oi_eth"] * s_oie +
        w["trends"] * s_tr +
        w["fng_now"] * s_fng_now +
        w["fng_ma14"] * s_fng14 +
        w["fng_ma30"] * s_fng30 +
        w["fng_persist"] * s_persist +
        w["pi"] * s_pi +
        w["rsi_btc2w"] * s_rsi_b +
        w["rsi_ethbtc2w"] * s_rsi_e +
        w["rsi_alt2w"] * s_rsi_a
    )
    return max(0.0, min(100.0, total * mult))

# ----------------------------
# Formatting snapshot
# ----------------------------


def glyph(value: Optional[float], warn: float, flag: float, higher_is_risk: bool = True) -> str:
    return col(tri_flag(value, warn, flag, higher_is_risk))


def build_text_snapshot(m: Dict[str, Any], profile: str) -> str:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    backtest_info = f"(as of {m.get('as_of')}) " if m.get("as_of") else ""

    # Market structure
    dom_val = m.get("btc_dom")
    dom_g = glyph(dom_val, THRESH["btc_dominance_warn"],
                  THRESH["btc_dominance_flag"], True)
    eth_btc_val = m.get("eth_btc")
    ethbtc_g = glyph(
        eth_btc_val, THRESH["eth_btc_warn"], THRESH["eth_btc_flag"], True)
    altcap_val = m.get("altcap_btc_ratio")
    altcap_g = glyph(
        altcap_val, THRESH["altcap_btc_warn"], THRESH["altcap_btc_flag"], True)

    # Derivatives
    fund_max = m.get("funding_max")
    fund_med = m.get("funding_median")
    fund_g_max = glyph(
        fund_max, THRESH["funding_warn"], THRESH["funding_flag"], True)
    fund_g_med = glyph(
        fund_med, THRESH["funding_warn"], THRESH["funding_flag"], True)

    oi_btc = m.get("oi_btc")
    oi_eth = m.get("oi_eth")
    oi_btc_g = glyph(oi_btc, THRESH["oi_btc_warn"],
                     THRESH["oi_btc_flag"], True)
    oi_eth_g = glyph(oi_eth, THRESH["oi_eth_warn"],
                     THRESH["oi_eth_flag"], True)

    # Sentiment
    trends = m.get("trends")
    trends_g = glyph(trends, THRESH["trends_warn"],
                     THRESH["trends_flag"], True)
    fng = m.get("fng") or {}
    fng_now = fng.get("now")
    fng14 = fng.get("ma14")
    fng30 = fng.get("ma30")
    fng_streak = fng.get("greed_streak")
    fng_pct = fng.get("greedy_pct30")

    fng_now_g = glyph(
        fng_now, THRESH["fng_warn"], THRESH["fng_flag"], True) if fng_now is not None else "🟡"
    fng14_g = glyph(fng14, THRESH["fng14_warn"],
                    THRESH["fng14_flag"], True) if fng14 is not None else "🟡"
    fng30_g = glyph(fng30, THRESH["fng30_warn"],
                    THRESH["fng30_flag"], True) if fng30 is not None else "🟡"

    days_warn = THRESH["fng_persist_days_warn"]
    days_flag = THRESH["fng_persist_days_flag"]
    pct_warn = THRESH["fng_persist_pct_warn"]
    pct_flag = THRESH["fng_persist_pct_flag"]
    persist_flag = "green"
    if (fng_streak is not None and fng_streak >= days_flag) or (fng_pct is not None and fng_pct >= pct_flag):
        persist_flag = "red"
    elif (fng_streak is not None and fng_streak >= days_warn) or (fng_pct is not None and fng_pct >= pct_warn):
        persist_flag = "yellow"
    persist_g = col(persist_flag)

    # Cycle & on-chain (Pi Cycle proximity)
    pi_p = m.get("pi_prox")
    pi_g = "🟡" if pi_p is None else (
        "🟢" if pi_p < 0.7 else ("🟡" if pi_p < 0.9 else "🔴"))
    pi_txt = "n/a" if pi_p is None else f"{pi_p*100:.2f}% of trigger (100% = cross)"

    # Momentum (2W) & Extensions (1W)
    rsi_btc2w = m.get("rsi_btc2w")
    rsi_ethbtc2w = m.get("rsi_ethbtc2w")
    rsi_alt2w = m.get("rsi_alt2w")
    rsi_btc2w_ma = m.get("rsi_btc2w_ma")
    rsi_alt2w_ma = m.get("rsi_alt2w_ma")

    rsi_btc_g = glyph(
        rsi_btc2w, THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"], True)
    rsi_ethbtc_g = glyph(
        rsi_ethbtc2w, THRESH["rsi_ethbtc2w_warn"], THRESH["rsi_ethbtc2w_flag"], True)
    rsi_alt_g = glyph(
        rsi_alt2w, THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"], True)

    k_btc2w, d_btc2w = m.get("k_btc2w"), m.get("d_btc2w")
    k_alt2w, d_alt2w = m.get("k_alt2w"), m.get("d_alt2w")

    btc_fib = m.get("btc_fib")
    alt_fib = m.get("alt_fib")

    # Composite certainty
    certainty = round(alt_top_certainty(m, profile))
    cert_flag = "green" if certainty < 40 else (
        "yellow" if certainty < 70 else "red")
    cert_g = col(cert_flag)

    lines: List[str] = []
    lines.append(
        f"📊 Crypto Market Snapshot — {now_iso} {backtest_info}".strip())
    lines.append(f"Profile: {profile}")
    if m.get("as_of"):
        lines.append(
            "_Backtest note: market cap, derivatives, and sentiment are omitted when assessing past dates; price-derived signals only._")
    lines.append("")
    lines.append("Market Structure")
    if dom_val is not None:
        lines.append(
            "• Bitcoin market share of total crypto: {} {:.2f}%  (warn ≥ {:.2f}%, flag ≥ {:.2f}%)".format(
                dom_g, dom_val *
                100.0, THRESH["btc_dominance_warn"] *
                100.0, THRESH["btc_dominance_flag"] * 100.0
            )
        )
    else:
        lines.append("• Bitcoin market share of total crypto: 🟡 n/a")
    if eth_btc_val is not None:
        lines.append("• Ether price relative to Bitcoin (ETH/BTC): {} {:.5f}  (warn ≥ {:.5f}, flag ≥ {:.5f})".format(
            ethbtc_g, eth_btc_val, THRESH["eth_btc_warn"], THRESH["eth_btc_flag"]
        ))
    else:
        lines.append("• Ether price relative to Bitcoin (ETH/BTC): 🟡 n/a")
    if altcap_val is not None:
        lines.append("• Altcoin market cap / Bitcoin market cap: {} {:.2f}  (warn ≥ {:.2f}, flag ≥ {:.2f})".format(
            altcap_g, altcap_val, THRESH["altcap_btc_warn"], THRESH["altcap_btc_flag"]
        ))
    else:
        lines.append("• Altcoin market cap / Bitcoin market cap: 🟡 n/a")

    lines.append("")
    lines.append("Derivatives")
    if fund_max is None and fund_med is None:
        lines.append("• Funding basket: 🟡 n/a")
    else:
        lines.append(
            "• Funding (basket: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT) — max: {} {} | median: {} {}  (warn ≥ {:.3f}%, flag ≥ {:.3f}%)".format(
                fund_g_max, pct_str(
                    fund_max, 3), fund_g_med, pct_str(fund_med, 3),
                THRESH['funding_warn'] * 100.0, THRESH['funding_flag'] * 100.0
            )
        )
    lines.append("• Bitcoin open interest (USD): {} {}  (warn ≥ {}, flag ≥ {})".format(
        oi_btc_g, to_human_dollar(oi_btc), to_human_dollar(
            THRESH['oi_btc_warn']), to_human_dollar(THRESH['oi_btc_flag'])
    ))
    lines.append("• Ether open interest (USD): {} {}  (warn ≥ {}, flag ≥ {})".format(
        oi_eth_g, to_human_dollar(oi_eth), to_human_dollar(
            THRESH['oi_eth_warn']), to_human_dollar(THRESH['oi_eth_flag'])
    ))

    lines.append("")
    lines.append("Sentiment")
    if trends is not None:
        lines.append("• Google Trends avg (7d; crypto/bitcoin/ethereum): {} {:.1f}  (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
            trends_g, trends, THRESH["trends_warn"], THRESH["trends_flag"]
        ))
    else:
        lines.append(
            "• Google Trends avg (7d; crypto/bitcoin/ethereum): 🟡 n/a")
    if fng_now is not None:
        lines.append("• Fear & Greed Index (overall crypto): {} {}  (warn ≥ {}, flag ≥ {})".format(
            fng_now_g, fng_now, THRESH["fng_warn"], THRESH["fng_flag"]
        ))
    else:
        lines.append("• Fear & Greed Index (overall crypto): 🟡 n/a")
    if fng14 is not None:
        lines.append("• Fear & Greed 14-day average: {} {:.1f}  (warn ≥ {}, flag ≥ {})".format(
            fng14_g, fng14, THRESH["fng14_warn"], THRESH["fng14_flag"]
        ))
    else:
        lines.append("• Fear & Greed 14-day average: 🟡 n/a")
    if fng30 is not None:
        lines.append("• Fear & Greed 30-day average: {} {:.1f}  (warn ≥ {}, flag ≥ {})".format(
            fng30_g, fng30, THRESH["fng30_warn"], THRESH["fng30_flag"]
        ))
    else:
        lines.append("• Fear & Greed 30-day average: 🟡 n/a")
    if (fng_streak is not None) and (fng_pct is not None):
        lines.append(
            "• Greed persistence: {} {} days in a row | {}% of last 30 days ≥ 70  (warn: days ≥ {} or pct ≥ {}%; flag: days ≥ {} or pct ≥ {}%)".format(
                persist_g, fng_streak, int((fng_pct or 0.0) * 100),
                days_warn, int(pct_warn * 100), days_flag, int(pct_flag * 100)
            )
        )
    else:
        lines.append("• Greed persistence: 🟡 n/a")

    lines.append("")
    lines.append("Cycle & On-Chain")
    lines.append(f"• Pi Cycle Top proximity: {pi_g} {pi_txt}")

    lines.append("")
    lines.append("Momentum (2W) & Extensions (1W)")
    if rsi_btc2w is not None:
        if rsi_btc2w_ma is not None:
            lines.append("• BTC RSI (2W): {} {:.1f} (MA {:.1f}) (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
                rsi_btc_g, rsi_btc2w, rsi_btc2w_ma, THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"]
            ))
        else:
            lines.append("• BTC RSI (2W): {} {:.1f} (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
                rsi_btc_g, rsi_btc2w, THRESH["rsi_btc2w_warn"], THRESH["rsi_btc2w_flag"]
            ))
    else:
        lines.append("• BTC RSI (2W): 🟡 n/a")

    if rsi_ethbtc2w is not None:
        lines.append("• ETH/BTC RSI (2W): {} {:.1f} (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
            rsi_ethbtc_g, rsi_ethbtc2w, THRESH["rsi_ethbtc2w_warn"], THRESH["rsi_ethbtc2w_flag"]
        ))
    else:
        lines.append("• ETH/BTC RSI (2W): 🟡 n/a")

    if rsi_alt2w is not None:
        if rsi_alt2w_ma is not None:
            lines.append("• ALT basket (equal-weight) RSI (2W): {} {:.1f} (MA {:.1f}) (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
                rsi_alt_g, rsi_alt2w, rsi_alt2w_ma, THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"]
            ))
        else:
            lines.append("• ALT basket (equal-weight) RSI (2W): {} {:.1f} (warn ≥ {:.1f}, flag ≥ {:.1f})".format(
                rsi_alt_g, rsi_alt2w, THRESH["rsi_alt2w_warn"], THRESH["rsi_alt2w_flag"]
            ))
    else:
        lines.append("• ALT basket (equal-weight) RSI (2W): 🟡 n/a")

    if (k_btc2w is not None) and (d_btc2w is not None):
        lines.append("• BTC Stoch RSI (2W) K/D: {:.2f}/{:.2f} (overbought ≥ 0.80; red = bearish cross from OB)".format(
            k_btc2w, d_btc2w
        ))
    else:
        lines.append("• BTC Stoch RSI (2W) K/D: 🟡 n/a")

    if (k_alt2w is not None) and (d_alt2w is not None):
        lines.append("• ALT basket Stoch RSI (2W) K/D: {:.2f}/{:.2f} (overbought ≥ 0.80; red = bearish cross from OB)".format(
            k_alt2w, d_alt2w
        ))
    else:
        lines.append("• ALT basket Stoch RSI (2W) K/D: 🟡 n/a")

    if btc_fib:
        _ext_btc, away_btc = btc_fib
        fib_g_btc = glyph(
            away_btc, THRESH["fib_warn"], THRESH["fib_flag"], False)
        lines.append("• BTC Fibonacci extension proximity: {} 1.272 @ {:.2f}% away (warn ≤ {:.1f}%, flag ≤ {:.1f}%)".format(
            fib_g_btc, away_btc *
            100.0, THRESH["fib_warn"] * 100.0, THRESH["fib_flag"] * 100.0
        ))
    else:
        lines.append("• BTC Fibonacci extension proximity: 🟡 n/a")

    if alt_fib:
        _ext_alt, away_alt = alt_fib
        fib_g_alt = glyph(
            away_alt, THRESH["fib_warn"], THRESH["fib_flag"], False)
        lines.append("• ALT basket Fibonacci proximity: {} 1.272 @ {:.2f}% away (warn ≤ {:.1f}%, flag ≤ {:.1f}%)".format(
            fib_g_alt, away_alt *
            100.0, THRESH["fib_warn"] * 100.0, THRESH["fib_flag"] * 100.0
        ))
    else:
        lines.append("• ALT basket Fibonacci proximity: 🟡 n/a")

    lines.append("")
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {cert_g} {certainty}/100 (yellow ≥ 40, red ≥ 70)")

    return "\n".join(lines)

# ----------------------------
# Telegram Handlers
# ----------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    if chat_id not in CHAT_PROFILE:
        CHAT_PROFILE[chat_id] = "moderate"
    msg = [
        "Hey! I’ll watch key cycle metrics and send updates here.",
        "",
        "Commands:",
        "/assess — fetch a fresh snapshot now",
        "/assess_at <YYYY-MM-DD> — backtest snapshot using price-derived signals available up to that date",
        "/subscribe — daily snapshot + alerts",
        "/unsubscribe — stop automated messages",
        "/setprofile <conservative|moderate|aggressive> — adjust risk focus",
        "/help — help & thresholds",
    ]
    await update.message.reply_text("\n".join(msg), disable_web_page_preview=True)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    t = [
        "I track market structure, funding & OI, sentiment (Google Trends & Fear/Greed with persistence),",
        "momentum (2-week RSI/Stoch RSI), Fibonacci extensions, and Pi Cycle Top proximity.",
        "",
        "Backtests: /assess_at YYYY-MM-DD uses only price-derived signals up to that date (no market-cap, derivatives, or F&G).",
        "",
        "Traffic lights:",
        "🟢 green: below warn (or above, where higher is good)",
        "🟡 yellow: between warn and flag",
        "🔴 red: beyond flag",
        "",
        "Set your profile with /setprofile conservative|moderate|aggressive.",
        "Use /subscribe to get a once-a-day snapshot plus periodic alerts.",
    ]
    await update.message.reply_text("\n".join(t), disable_web_page_preview=True)


async def cmd_setprofile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        arg = (context.args[0].strip().lower() if context.args else "")
    except Exception:
        arg = ""
    if arg not in RISK_PROFILES:
        await update.message.reply_text("Usage: /setprofile <conservative|moderate|aggressive>")
        return
    CHAT_PROFILE[chat_id] = arg
    await update.message.reply_text(f"Profile set to: {arg}")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.add(chat_id)
    if chat_id not in CHAT_PROFILE:
        CHAT_PROFILE[chat_id] = "moderate"
    await update.message.reply_text("Subscribed. You’ll receive daily snapshots and periodic alerts here.")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBERS.discard(chat_id)
    await update.message.reply_text("Unsubscribed. You’ll stop receiving automated messages.")


async def build_and_send_snapshot(app: Application, chat_id: int, profile: str, as_of: Optional[datetime] = None):
    try:
        m = await build_metrics(as_of=as_of)
        text = build_text_snapshot(m, profile)
        await app.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True)
    except Exception as e:
        log.exception("snapshot send failed")
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Could not fetch metrics right now: {e}")


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    profile = CHAT_PROFILE.get(chat_id, "moderate")
    await build_and_send_snapshot(context.application, chat_id, profile)


async def cmd_assess_at(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    profile = CHAT_PROFILE.get(chat_id, "moderate")
    if not context.args:
        await update.message.reply_text("Usage: /assess_at YYYY-MM-DD (UTC)")
        return
    dt = parse_date(context.args[0])
    if not dt:
        await update.message.reply_text("Invalid date. Use YYYY-MM-DD (UTC).")
        return
    await build_and_send_snapshot(context.application, chat_id, profile, as_of=dt)

# ----------------------------
# Scheduler Jobs
# ----------------------------


async def job_push_summary(app: Application):
    if not SUBSCRIBERS:
        return
    for chat_id in list(SUBSCRIBERS):
        profile = CHAT_PROFILE.get(chat_id, "moderate")
        await build_and_send_snapshot(app, chat_id, profile)


async def job_push_alerts(app: Application):
    if not SUBSCRIBERS:
        return
    for chat_id in list(SUBSCRIBERS):
        profile = CHAT_PROFILE.get(chat_id, "moderate")
        await build_and_send_snapshot(app, chat_id, profile)

# ----------------------------
# Health server
# ----------------------------


async def health_handler(_request):
    return web.json_response({
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "subscribers": len(SUBSCRIBERS),
    })


async def start_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=8080)
    await site.start()
    log.info("Health server listening on :8080")
    return runner, site

# ----------------------------
# Main
# ----------------------------


async def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")

    # Telegram app
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("setprofile", cmd_setprofile))
    application.add_handler(CommandHandler("subscribe", cmd_subscribe))
    application.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    application.add_handler(CommandHandler("assess", cmd_assess))
    application.add_handler(CommandHandler("assess_at", cmd_assess_at))

    # Health server
    runner, _site = await start_health_server()

    # Scheduler
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(lambda: asyncio.create_task(job_push_summary(application)),
                      trigger="cron", hour=SUMMARY_UTC_HOUR, minute=0)
    scheduler.add_job(lambda: asyncio.create_task(job_push_alerts(application)),
                      trigger="cron", minute=ALERT_CRON_MINUTES)
    scheduler.start()
    log.info("Scheduler started")

    # Start polling (PTB 21 pattern)
    # Remove webhook (safety), then start polling without blocking the event loop
    try:
        await application.bot.delete_webhook(drop_pending_updates=False)
        log.info("Deleted webhook (if any); switching to long polling.")
    except Exception:
        pass

    await application.initialize()
    await application.start()
    # PTB 21: Updater is optional; if present, start it
    if application.updater:
        await application.updater.start_polling()
    log.info("Application started; polling for updates.")

    log.info("Bot running. Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if application.updater:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        scheduler.shutdown(wait=False)
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
