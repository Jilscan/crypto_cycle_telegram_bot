import os
import asyncio
import logging
from datetime import datetime, timezone, time as dtime
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler, ContextTypes
)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s"
)
log = logging.getLogger("crypto-cycle-bot")

# -----------------------------
# Emoji helpers
# -----------------------------
GREEN = "🟢"
YELLOW = "🟡"
RED = "🔴"


def tri_color(value: Optional[float], warn: Optional[float], flag: Optional[float], reverse: bool = False) -> str:
    """
    Tri-color evaluation:
    - reverse=False: green < warn, yellow [warn, flag), red >= flag
    - reverse=True:  green > warn, yellow (flag, warn], red <= flag
    None -> yellow (unknown)
    """
    if value is None or warn is None or flag is None:
        return YELLOW
    if not reverse:
        if value < warn:
            return GREEN
        if value < flag:
            return YELLOW
        return RED
    else:
        if value > warn:
            return GREEN
        if value > flag:
            return YELLOW
        return RED


def fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x >= 1e12:
        return f"${x/1e12:.3f}T"
    if x >= 1e9:
        return f"${x/1e9:.3f}B"
    if x >= 1e6:
        return f"${x/1e6:.3f}M"
    return f"${x:,.0f}"


def fmt_pct(x: Optional[float], decimals=2) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.{decimals}f}%"


def pct_away(price: float, target: float) -> float:
    if target == 0:
        return 0.0
    return abs(price - target) / abs(target)

# -----------------------------
# Math: RSI / StochRSI
# -----------------------------


def rsi(series: List[float], period: int = 14) -> pd.Series:
    s = pd.Series(series, dtype=float)
    if len(s) < period + 5:
        return pd.Series([np.nan] * len(s))
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False,
                         min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0.0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out


def stoch_rsi_from_rsi(rsi_series: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    r = rsi_series.copy()
    if len(r) < period + smooth_k + smooth_d:
        return pd.Series([np.nan] * len(r)), pd.Series([np.nan] * len(r))
    min_r = r.rolling(period, min_periods=period).min()
    max_r = r.rolling(period, min_periods=period).max()
    denom = (max_r - min_r).replace(0.0, np.nan)
    k = (r - min_r) / denom
    k = k.clip(lower=0.0, upper=1.0)
    k_s = k.rolling(smooth_k, min_periods=smooth_k).mean()
    d_s = k_s.rolling(smooth_d, min_periods=smooth_d).mean()
    return k_s, d_s


def two_week_from_weekly(weekly_closes: List[float]) -> List[float]:
    """
    Build a 2-week series from weekly closes by taking every 2nd week's close
    (the closing price of each two-week window).
    """
    if len(weekly_closes) < 4:
        return []
    return [weekly_closes[i] for i in range(1, len(weekly_closes), 2)]


def simple_sma(values: List[float], period: int) -> pd.Series:
    s = pd.Series(values, dtype=float)
    return s.rolling(period, min_periods=period).mean()

# -----------------------------
# Data client with backups
# -----------------------------


class DataClient:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout, headers={
                                        "User-Agent": "crypto-cycle-bot/1.0"})
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client:
            await self.client.aclose()

    # ------------ OKX ------------
    async def okx_get(self, path: str, params: Dict[str, str]) -> Dict:
        url = f"https://www.okx.com/api/v5{path}"
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def okx_ticker_last(self, inst_id: str) -> Optional[float]:
        try:
            j = await self.okx_get("/market/ticker", {"instId": inst_id})
            data = (j.get("data") or [{}])[0]
            return float(data.get("last", "nan"))
        except Exception:
            return None

    async def okx_funding_rate(self, inst_id: str) -> Optional[float]:
        try:
            j = await self.okx_get("/public/funding-rate", {"instId": inst_id})
            data = (j.get("data") or [{}])[0]
            fr = data.get("fundingRate")
            return float(fr) if fr is not None else None
        except Exception:
            return None

    async def okx_oi_usd(self, inst_id: str) -> Optional[float]:
        """
        Priority:
          1) oiValue (direct USD)
          2) oiCcy for -USDT-SWAP (already in USDT)
          3) oi * last_price
        """
        try:
            j = await self.okx_get("/public/open-interest", {"instType": "SWAP", "instId": inst_id})
            d = (j.get("data") or [{}])[0]
            if d.get("oiValue") is not None:
                return float(d["oiValue"])
            if d.get("oiCcy") is not None and inst_id.endswith("-USDT-SWAP"):
                return float(d["oiCcy"])
            oi = d.get("oi")
            if oi is not None:
                last = await self.okx_ticker_last(inst_id)
                if last is not None:
                    return float(oi) * float(last)
        except Exception:
            pass
        return None

    async def okx_candles(self, inst_id: str, bar: str, limit: int) -> List[Tuple[int, float]]:
        """
        Returns ascending [(timestamp_ms, close), ...]
        """
        try:
            j = await self.okx_get("/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)})
            rows = j.get("data") or []
            out = []
            for row in reversed(rows):  # newest-first -> oldest-first
                ts_ms = int(row[0])
                close = float(row[4])
                out.append((ts_ms, close))
            return out
        except Exception:
            return []

    # --------- Coinbase (backup) ---------
    async def coinbase_get(self, path: str, params: Dict[str, str]) -> List:
        url = f"https://api.exchange.coinbase.com{path}"
        r = await self.client.get(url, params=params, headers={"Accept": "application/json"})
        r.raise_for_status()
        return r.json()

    async def coinbase_candles(self, product_id: str, granularity_sec: int, limit: int = 300) -> List[Tuple[int, float]]:
        """
        Coinbase returns arrays [time, low, high, open, close, volume] in descending time.
        We return ascending [(ts_ms, close), ...]
        """
        try:
            data = await self.coinbase_get(f"/products/{product_id}/candles", {"granularity": str(granularity_sec)})
            data_sorted = sorted(data, key=lambda x: x[0])
            out = [(int(ts)*1000, float(close))
                   for ts, low, high, open_, close, vol in data_sorted][-limit:]
            return out
        except Exception:
            return []

    # --------- Kraken (backup) ----------
    async def kraken_get(self, path: str, params: Dict[str, str]) -> Dict:
        url = f"https://api.kraken.com/0/public{path}"
        r = await self.client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    async def kraken_candles(self, pair: str, interval_min: int, limit: int = 720) -> List[Tuple[int, float]]:
        """
        Kraken OHLC: result[pair] -> [ [time, open, high, low, close, vwap, volume, count], ... ]
        Return ascending [(ts_ms, close)]
        """
        try:
            j = await self.kraken_get("/OHLC", {"pair": pair, "interval": str(interval_min)})
            res = j.get("result", {})
            keys = [k for k in res.keys() if k != "last"]
            if not keys:
                return []
            arr = res[keys[0]]
            out = [(int(row[0])*1000, float(row[4])) for row in arr][-limit:]
            return out
        except Exception:
            return []

    # --------- Convenience ----------
    async def btc_daily(self, limit: int = 1500) -> List[Tuple[int, float]]:
        d = await self.okx_candles("BTC-USDT", "1D", limit)
        if len(d) >= 360:
            return d
        d = await self.coinbase_candles("BTC-USD", 86400, limit)
        if len(d) >= 360:
            return d
        d = await self.kraken_candles("XBTUSD", 1440, limit)
        return d

    async def weekly_closes(self, inst_id_okx: str, cb_product: Optional[str], kr_pair: Optional[str], limit_w: int = 400) -> List[float]:
        w = await self.okx_candles(inst_id_okx, "1W", limit_w)
        if len(w) >= 60:
            return [c for _, c in w]
        if cb_product:
            d = await self.coinbase_candles(cb_product, 86400, 1500)
            if len(d) >= 90:
                closes = [c for _, c in d]
                wk = []
                for i in range(6, len(closes), 7):
                    wk.append(closes[i])
                if len(wk) >= 60:
                    return wk
        if kr_pair:
            d = await self.kraken_candles(kr_pair, 1440, 1500)
            if len(d) >= 90:
                closes = [c for _, c in d]
                wk = []
                for i in range(6, len(closes), 7):
                    wk.append(closes[i])
                return wk
        return []

    async def eth_btc_last(self) -> Optional[float]:
        last = await self.okx_ticker_last("ETH-BTC")
        if last is not None and not np.isnan(last):
            return last
        try:
            eth = await self.coinbase_candles("ETH-USD", 86400, 2)
            btc = await self.coinbase_candles("BTC-USD", 86400, 2)
            if eth and btc:
                return eth[-1][1] / btc[-1][1]
        except Exception:
            pass
        try:
            eth = await self.kraken_candles("ETHUSD", 1440, 2)
            btc = await self.kraken_candles("XBTUSD", 1440, 2)
            if eth and btc:
                return eth[-1][1] / btc[-1][1]
        except Exception:
            pass
        return None

    async def btc_dominance_and_altcap_ratio(self) -> Tuple[Optional[float], Optional[float]]:
        """
        CoinGecko global:
          - market_cap_percentage.btc (%)
          - total_market_cap.usd
        Altcap/BTC ≈ (total - btc) / btc
        """
        try:
            r = await self.client.get("https://api.coingecko.com/api/v3/global")
            r.raise_for_status()
            j = r.json()
            data = j.get("data", {})
            btc_pct = float(
                data.get("market_cap_percentage", {}).get("btc", np.nan))
            total_usd = float(
                data.get("total_market_cap", {}).get("usd", np.nan))
            if not np.isnan(btc_pct) and not np.isnan(total_usd) and btc_pct > 0:
                btc_mcap = total_usd * (btc_pct / 100.0)
                altcap_ratio = (total_usd - btc_mcap) / \
                    btc_mcap if btc_mcap > 0 else None
            else:
                altcap_ratio = None
            return (btc_pct, altcap_ratio)
        except Exception:
            return (None, None)

# -----------------------------
# Sentiment sources
# -----------------------------


async def google_trends_avg7() -> Optional[float]:
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=0)
        pytrends.build_payload(
            ["crypto", "bitcoin", "ethereum"], timeframe="now 7-d", geo="")
        df = pytrends.interest_over_time()
        if df.empty:
            return None
        vals = df[["crypto", "bitcoin", "ethereum"]].mean(axis=1)
        return float(vals.mean())
    except Exception as e:
        log.warning("google_trends_avg7 failed: %s", e)
        return None


async def fear_greed() -> Dict[str, Optional[float]]:
    """
    Returns dict with keys: value (today, 0-100), ma14, ma30, streak_ge70_days, pct30_ge70
    """
    out = {"value": None, "ma14": None, "ma30": None,
           "streak_ge70_days": None, "pct30_ge70": None}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r1 = await c.get("https://api.alternative.me/fng/?limit=1")
            r1.raise_for_status()
            v = r1.json().get("data", [{}])[0]
            today = float(v.get("value", "nan"))
            out["value"] = today if not np.isnan(today) else None

            r30 = await c.get("https://api.alternative.me/fng/?limit=60")
            r30.raise_for_status()
            arr = r30.json().get("data", [])
            vals = [float(d.get("value", "nan")) for d in arr]
            vals = [x for x in vals if not np.isnan(x)]
            if len(vals) >= 30:
                last30 = vals[:30]
                out["ma30"] = float(np.mean(last30))
                out["pct30_ge70"] = float(
                    np.mean([1.0 if x >= 70 else 0.0 for x in last30]))
            if len(vals) >= 14:
                out["ma14"] = float(np.mean(vals[:14]))
            streak = 0
            for x in vals:
                if x >= 70:
                    streak += 1
                else:
                    break
            out["streak_ge70_days"] = float(streak)
    except Exception as e:
        log.warning("fear_greed failed: %s", e)
    return out

# -----------------------------
# Momentum blocks
# -----------------------------
ALT_BASKET_OKX = ["SOL-USDT", "XRP-USDT", "DOGE-USDT",
                  "ADA-USDT", "BNB-USDT", "AVAX-USDT", "LINK-USDT", "MATIC-USDT"]
CB_MAP = {
    "SOL-USDT": "SOL-USD",
    "XRP-USDT": "XRP-USD",
    "DOGE-USDT": "DOGE-USD",
    "ADA-USDT": "ADA-USD",
    "BNB-USDT": None,
    "AVAX-USDT": "AVAX-USD",
    "LINK-USDT": "LINK-USD",
    "MATIC-USDT": "MATIC-USD",
}
KR_MAP = {
    "SOL-USDT": "SOLUSD",
    "XRP-USDT": "XRPUSD",
    "DOGE-USDT": "DOGEUSD",
    "ADA-USDT": "ADAUSD",
    "BNB-USDT": None,
    "AVAX-USDT": "AVAXUSD",
    "LINK-USDT": "LINKUSD",
    "MATIC-USDT": "MATICUSD",
}


async def weekly_closes_with_backups(dc: DataClient, inst_okx: str) -> List[float]:
    cb = CB_MAP.get(inst_okx)
    kr = KR_MAP.get(inst_okx)
    return await dc.weekly_closes(inst_okx, cb, kr, limit_w=400)


async def alt_basket_weekly_index(dc: DataClient) -> List[float]:
    series = []
    for sym in ALT_BASKET_OKX:
        w = await weekly_closes_with_backups(dc, sym)
        if len(w) >= 60:
            series.append(np.array(w, dtype=float))
    if not series:
        return []
    min_len = min(len(s) for s in series)
    if min_len < 30:
        return []
    series = [s[-min_len:] for s in series]
    norm = [100.0 * s / s[0] for s in series]
    idx = np.mean(np.vstack(norm), axis=0)
    return list(idx)


def fib_extension_proximity(weekly_closes: List[float], lookback_weeks: int = 52, ext: float = 1.272) -> Optional[float]:
    if len(weekly_closes) < lookback_weeks + 5:
        return None
    closes = np.array(weekly_closes[-(lookback_weeks+1):], dtype=float)
    last = closes[-1]
    swing_low = float(np.min(closes[:-1]))
    swing_high = float(np.max(closes[:-1]))
    if swing_high <= swing_low:
        return None
    ext_target = swing_high + (swing_high - swing_low) * (ext - 1.0)
    return pct_away(last, ext_target)

# -----------------------------
# Pi Cycle Top proximity
# -----------------------------


async def pi_cycle_proximity(dc: DataClient) -> Optional[float]:
    """
    Daily: SMA111 vs 2*SMA350; proximity (%) to cross (0% at exact cross).
    If daily insufficient, fallback to weekly with SMA16 vs 2*SMA50 (≈).
    """
    d = await dc.btc_daily(1500)
    if len(d) >= 360:
        closes = pd.Series([c for _, c in d], dtype=float)
        sma111 = closes.rolling(111, min_periods=111).mean()
        sma350 = closes.rolling(350, min_periods=350).mean()
        if not sma111.dropna().empty and not sma350.dropna().empty:
            a = float(sma111.iloc[-1])
            b = float(2.0 * sma350.iloc[-1])
            denom = max(1e-9, abs(b))
            return abs(a - b) / denom * 100.0
    w = await dc.okx_candles("BTC-USDT", "1W", 400)
    if len(w) >= 60:
        closes = pd.Series([c for _, c in w], dtype=float)
        sma16 = closes.rolling(16, min_periods=16).mean()
        sma50 = closes.rolling(50, min_periods=50).mean()
        if not sma16.dropna().empty and not sma50.dropna().empty:
            a = float(sma16.iloc[-1])
            b = float(2.0 * sma50.iloc[-1])
            denom = max(1e-9, abs(b))
            return abs(a - b) / denom * 100.0
    return None

# -----------------------------
# Composite scoring
# -----------------------------


def severity_from_thresholds(value: Optional[float], warn: float, flag: float, reverse: bool = False) -> float:
    """
    Convert a metric to 0..1 severity using warn/flag. 0 best (green), 1 worst (red).
    """
    if value is None:
        return 0.4  # unknown -> mild yellow
    v = value
    if not reverse:
        if v < warn:
            return 0.0
        if v >= flag:
            return 1.0
        return (v - warn) / (flag - warn)
    else:
        if v > warn:
            return 0.0
        if v <= flag:
            return 1.0
        return (warn - v) / (warn - flag)


def color_from_certainty(score100: int) -> str:
    if score100 >= 70:
        return RED
    if score100 >= 40:
        return YELLOW
    return GREEN

# -----------------------------
# Build snapshot
# -----------------------------


async def build_metrics(profile: str) -> Dict:
    t0 = datetime.now(timezone.utc).isoformat()
    out: Dict = {"timestamp": t0, "profile": profile}

    async with DataClient() as dc:
        # Market structure
        dom, alt_ratio = await dc.btc_dominance_and_altcap_ratio()
        ethbtc = await dc.eth_btc_last()

        out["btc_dominance_pct"] = dom  # %
        out["altcap_btc_ratio"] = alt_ratio
        out["eth_btc"] = ethbtc

        # Derivatives (OKX)
        basket = ["BTC-USDT-SWAP", "ETH-USDT-SWAP",
                  "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP"]
        funding_rates: List[Tuple[str, Optional[float]]] = []
        for inst in basket:
            fr = await dc.okx_funding_rate(inst)
            funding_rates.append((inst, fr))
            await asyncio.sleep(0.1)
        out["funding_rates"] = funding_rates
        out["oi_btc_usd"] = await dc.okx_oi_usd("BTC-USDT-SWAP")
        out["oi_eth_usd"] = await dc.okx_oi_usd("ETH-USDT-SWAP")

        # Sentiment
        out["trends_avg7"] = await google_trends_avg7()
        fng = await fear_greed()
        out["fng"] = fng

        # Cycle: Pi
        out["pi_prox"] = await pi_cycle_proximity(dc)

        # Momentum (weekly & 2W)
        btc_w = await dc.weekly_closes("BTC-USDT", "BTC-USD", "XBTUSD", 400)

        # ETH/BTC weekly closes (fallback synth)
        ethbtc_w = await dc.weekly_closes("ETH-BTC", None, None, 400)
        if not ethbtc_w:
            eth_w = await dc.weekly_closes("ETH-USDT", "ETH-USD", "ETHUSD", 400)
            if eth_w and btc_w and len(eth_w) == len(btc_w):
                ethbtc_w = list(np.array(eth_w, dtype=float) /
                                np.array(btc_w, dtype=float))

        # Alt basket weekly index
        alt_idx_w = await alt_basket_weekly_index(dc)

        def rsi_block(closes_w: List[float]) -> Dict[str, Optional[float]]:
            if len(closes_w) < 40:
                return {"rsi": None, "rsi_ma": None, "k": None, "d": None}
            closes_2w = two_week_from_weekly(closes_w)
            if len(closes_2w) < 40:
                return {"rsi": None, "rsi_ma": None, "k": None, "d": None}
            r = rsi(closes_2w, period=14)
            rsi_ma = r.rolling(9, min_periods=9).mean()
            k, dline = stoch_rsi_from_rsi(r, period=14, smooth_k=3, smooth_d=3)
            return {
                "rsi": float(r.iloc[-1]) if not np.isnan(r.iloc[-1]) else None,
                "rsi_ma": float(rsi_ma.iloc[-1]) if not np.isnan(rsi_ma.iloc[-1]) else None,
                "k": float(k.iloc[-1]) if not np.isnan(k.iloc[-1]) else None,
                "d": float(dline.iloc[-1]) if not np.isnan(dline.iloc[-1]) else None,
            }

        out["btc_mom_2w"] = rsi_block(btc_w) if btc_w else {
            "rsi": None, "rsi_ma": None, "k": None, "d": None}
        out["ethbtc_mom_2w"] = rsi_block(ethbtc_w) if ethbtc_w else {
            "rsi": None, "rsi_ma": None, "k": None, "d": None}
        out["alt_idx_mom_2w"] = rsi_block(alt_idx_w) if alt_idx_w else {
            "rsi": None, "rsi_ma": None, "k": None, "d": None}

        # Fibonacci proximity (weekly)
        out["fib_btc_1272"] = fib_extension_proximity(
            btc_w, 52, 1.272) if btc_w else None
        out["fib_alt_1272"] = fib_extension_proximity(
            alt_idx_w, 52, 1.272) if alt_idx_w else None

        # Prices for fallback formatting (not strictly needed here)
        out["btc_usd_last"] = await dc.okx_ticker_last("BTC-USDT") or (await dc.okx_ticker_last("BTC-USD")) or None
        out["eth_usd_last"] = await dc.okx_ticker_last("ETH-USDT") or (await dc.okx_ticker_last("ETH-USD")) or None

    return out


def build_text(metrics: Dict) -> Tuple[str, int]:
    m = metrics
    profile = m.get("profile", "moderate")
    ts_iso = m.get("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        ts_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        ts_disp = ts_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        ts_disp = ts_iso

    lines: List[str] = []
    lines.append(f"📊 Crypto Market Snapshot — {ts_disp}")
    lines.append(f"Profile: {profile}")
    lines.append("")

    # ---------- Market Structure ----------
    lines.append("Market Structure")
    dom = m.get("btc_dominance_pct")
    dom_col = tri_color(dom, warn=48.0, flag=60.0)
    lines.append(
        f"• Bitcoin market share of total crypto: {dom_col} {f'{dom:.2f}%' if dom is not None else 'n/a'}  (warn ≥ 48.00%, flag ≥ 60.00%)")

    ethbtc = m.get("eth_btc")
    ethbtc_col = tri_color(ethbtc, warn=0.072, flag=0.09)
    lines.append(
        f"• Ether price relative to Bitcoin (ETH/BTC): {ethbtc_col} {f'{ethbtc:.5f}' if ethbtc is not None else 'n/a'}  (warn ≥ 0.07200, flag ≥ 0.09000)")

    alt_ratio = m.get("altcap_btc_ratio")
    alt_col = tri_color(alt_ratio, warn=1.44, flag=1.80)
    lines.append(
        f"• Altcoin market cap / Bitcoin market cap: {alt_col} {f'{alt_ratio:.2f}' if alt_ratio is not None else 'n/a'}  (warn ≥ 1.44, flag ≥ 1.80)")
    lines.append("")

    # ---------- Derivatives ----------
    lines.append("Derivatives")
    frs: List[Tuple[str, Optional[float]]] = m.get("funding_rates") or []
    vals = [x for _, x in frs if x is not None]
    fr_max = max(vals) if vals else None
    fr_med = float(np.median(vals)) if vals else None
    fr_col_max = tri_color(fr_max, warn=0.0008, flag=0.0010)  # 0.08% / 0.10%
    fr_col_med = tri_color(fr_med, warn=0.0008, flag=0.0010)
    basket_names = ", ".join([s.replace("-USDT-SWAP", "USDT")
                             for s, _ in frs]) or "n/a"
    lines.append(
        "• Funding (basket: " + basket_names +
        f") — max: {fr_col_max} {fmt_pct(fr_max)} | median: {fr_col_med} {fmt_pct(fr_med)}  (warn ≥ 0.080%, flag ≥ 0.100%)"
    )
    if frs:
        top3 = sorted([(s, abs(v) if v is not None else -1.0, v)
                      for s, v in frs], key=lambda x: x[1], reverse=True)[:3]
        pretty = []
        for s, _, v in top3:
            pretty.append(
                f"{s.replace('-USDT-SWAP','USDT')} {fmt_pct(v) if v is not None else 'n/a'}")
        lines.append(f"  Top-3 funding extremes: " + ", ".join(pretty))
    oi_btc = m.get("oi_btc_usd")
    oi_eth = m.get("oi_eth_usd")
    oi_btc_col = tri_color(oi_btc, warn=16e9, flag=20e9)
    oi_eth_col = tri_color(oi_eth, warn=6.4e9, flag=8.0e9)
    lines.append(
        f"• Bitcoin open interest (USD): {oi_btc_col} {fmt_usd(oi_btc)}  (warn ≥ $16.000B, flag ≥ $20.000B)")
    lines.append(
        f"• Ether open interest (USD): {oi_eth_col} {fmt_usd(oi_eth)}  (warn ≥ $6.400B, flag ≥ $8.000B)")
    lines.append("")

    # ---------- Sentiment ----------
    lines.append("Sentiment")
    trends = m.get("trends_avg7")
    trends_col = tri_color(trends, warn=60.0, flag=75.0)
    lines.append(
        f"• Google Trends avg (7d; crypto/bitcoin/ethereum): {trends_col} {f'{trends:.1f}' if trends is not None else 'n/a'}  (warn ≥ 60.0, flag ≥ 75.0)")

    fng = m.get("fng") or {}
    fng_today = fng.get("value")
    fng14 = fng.get("ma14")
    fng30 = fng.get("ma30")
    streak = fng.get("streak_ge70_days")
    pct30 = fng.get("pct30_ge70")

    fng_col = tri_color(fng_today, warn=56.0, flag=70.0)
    fng14_col = tri_color(fng14, warn=56.0, flag=70.0)
    fng30_col = tri_color(fng30, warn=52.0, flag=65.0)
    persist_text = "n/a"
    persist_col = YELLOW
    if streak is not None and pct30 is not None:
        persist_col = tri_color(max(streak, pct30 or 0.0), warn=8.0, flag=10.0)
        persist_text = f"{int(streak)} days in a row | {int(round((pct30 or 0.0)*100))}% of last 30 days ≥ 70"

    lines.append(
        f"• Fear & Greed Index (overall crypto): {fng_col} {int(fng_today) if fng_today is not None else 'n/a'}  (warn ≥ 56, flag ≥ 70)")
    lines.append(
        f"• Fear & Greed 14-day average: {fng14_col} {f'{fng14:.1f}' if fng14 is not None else 'n/a'}  (warn ≥ 56, flag ≥ 70)")
    lines.append(
        f"• Fear & Greed 30-day average: {fng30_col} {f'{fng30:.1f}' if fng30 is not None else 'n/a'}  (warn ≥ 52, flag ≥ 65)")
    lines.append(
        f"• Greed persistence: {persist_col} {persist_text}  (warn: days ≥ 8 or pct ≥ 48%; flag: days ≥ 10 or pct ≥ 60%)")
    lines.append("")

    # ---------- Cycle & On-Chain ----------
    lines.append("Cycle & On-Chain")
    pi = m.get("pi_prox")
    if pi is None:
        lines.append("• Pi Cycle Top proximity: 🟡 n/a")
    else:
        pi_col = tri_color(pi, warn=8.0, flag=3.0, reverse=True)
        lines.append(
            f"• Pi Cycle Top proximity: {pi_col} {pi:.2f}% of trigger (100% = cross)")
    lines.append("")

    # ---------- Momentum (2W) & Extensions (1W) ----------
    lines.append("Momentum (2W) & Extensions (1W)")

    def mom_line(name: str, block: Dict, warn_rsi: float, flag_rsi: float) -> List[str]:
        r = block.get("rsi")
        rma = block.get("rsi_ma")
        k = block.get("k")
        dline = block.get("d")
        col_r = tri_color(r, warn=warn_rsi, flag=flag_rsi)
        kd_txt = f"{f'{k:.2f}' if k is not None else 'n/a'}/{f'{dline:.2f}' if dline is not None else 'n/a'}"
        return [
            f"• {name} RSI (2W): {col_r} {f'{r:.1f}' if r is not None else 'n/a'}{f' (MA {rma:.1f})' if rma is not None else ''} (warn ≥ {warn_rsi:.1f}, flag ≥ {flag_rsi:.1f})",
            f"• {name} Stoch RSI (2W) K/D: {kd_txt} (overbought ≥ 0.80; red = bearish cross from OB)",
        ]

    lines += mom_line("BTC", m.get("btc_mom_2w", {}),
                      warn_rsi=60.0, flag_rsi=70.0)
    lines += mom_line("ETH/BTC", m.get("ethbtc_mom_2w", {}),
                      warn_rsi=55.0, flag_rsi=65.0)
    lines += mom_line("ALT basket (equal-weight)",
                      m.get("alt_idx_mom_2w", {}), warn_rsi=65.0, flag_rsi=75.0)

    fib_btc = m.get("fib_btc_1272")
    fib_alt = m.get("fib_alt_1272")
    fib_btc_col = tri_color(fib_btc, warn=0.03, flag=0.015,
                            reverse=True) if fib_btc is not None else YELLOW
    fib_alt_col = tri_color(fib_alt, warn=0.03, flag=0.015,
                            reverse=True) if fib_alt is not None else YELLOW
    lines.append(
        f"• BTC Fibonacci extension proximity: {fib_btc_col} 1.272 @ {f'{fib_btc*100:.2f}%' if fib_btc is not None else 'n/a'} away (warn ≤ 3.0%, flag ≤ 1.5%)")
    lines.append(
        f"• ALT basket Fibonacci proximity: {fib_alt_col} 1.272 @ {f'{fib_alt*100:.2f}%' if fib_alt is not None else 'n/a'} away (warn ≤ 3.0%, flag ≤ 1.5%)")
    lines.append("")

    # ---------- Composite scoring ----------
    # (label, raw_severity_0..100, weight_0..1)
    subs: List[Tuple[str, float, float]] = []
    subs.append(("altcap_vs_btc", severity_from_thresholds(
        alt_ratio, 1.44, 1.80) * 100, 0.08))
    subs.append(("eth_btc",       severity_from_thresholds(
        ethbtc, 0.072, 0.09) * 100, 0.08))

    frs_list = m.get("funding_rates") or []
    vals = [x for _, x in frs_list if x is not None]
    fr_max_val = max(vals) if vals else None
    subs.append(("funding_max",   severity_from_thresholds(
        fr_max_val, 0.0008, 0.0010) * 100, 0.10))
    subs.append(("OI_BTC",        severity_from_thresholds(
        m.get("oi_btc_usd"), 16e9, 20e9) * 100, 0.08))
    subs.append(("OI_ETH",        severity_from_thresholds(
        m.get("oi_eth_usd"), 6.4e9, 8.0e9) * 100, 0.06))

    subs.append(("Trends",        severity_from_thresholds(
        m.get("trends_avg7"), 60.0, 75.0) * 100, 0.05))
    fng_d = m.get("fng") or {}
    subs.append(("F&G today",     severity_from_thresholds(
        fng_d.get("value"), 56.0, 70.0) * 100, 0.06))
    subs.append(("F&G 14d",       severity_from_thresholds(
        fng_d.get("ma14"), 56.0, 70.0) * 100, 0.06))
    subs.append(("F&G 30d",       severity_from_thresholds(
        fng_d.get("ma30"), 52.0, 65.0) * 100, 0.08))
    streak = fng_d.get("streak_ge70_days")
    pct30 = fng_d.get("pct30_ge70")
    pers_raw = None
    if streak is not None and pct30 is not None:
        s_days = severity_from_thresholds(streak, 8.0, 10.0)
        s_pct = severity_from_thresholds(pct30, 0.48, 0.60)
        pers_raw = max(s_days, s_pct) * 100
    subs.append(
        ("F&G persistence", pers_raw if pers_raw is not None else 40.0, 0.08))

    subs.append(("Pi proximity",   severity_from_thresholds(
        m.get("pi_prox"), 8.0, 3.0, reverse=True) * 100, 0.07))

    def add_mom(name: str, block: Dict, warn: float, flag: float, weight: float):
        r = (block or {}).get("rsi")
        subs.append((name, severity_from_thresholds(
            r, warn, flag) * 100, weight))
    add_mom("RSI_BTC_2W",    m.get("btc_mom_2w", {}),     60.0, 70.0, 0.06)
    add_mom("RSI_ETHBTC_2W", m.get("ethbtc_mom_2w", {}),  55.0, 65.0, 0.04)
    add_mom("RSI_ALT_2W",    m.get("alt_idx_mom_2w", {}), 65.0, 75.0, 0.06)

    subs.append(("Fib_BTC_1.272", severity_from_thresholds(
        m.get("fib_btc_1272"), 0.03, 0.015, reverse=True) * 100, 0.04))
    subs.append(("Fib_ALT_1.272", severity_from_thresholds(
        m.get("fib_alt_1272"), 0.03, 0.015, reverse=True) * 100, 0.04))

    mult = 1.0 if profile == "moderate" else (
        0.8 if profile == "conservative" else 1.2)
    total = 0.0
    wsum = 0.0
    for _, raw, w in subs:
        total += (raw / 100.0) * (w * mult)
        wsum += (w * mult)
    certainty = int(round(100.0 * (total / wsum))) if wsum > 0 else 0
    cert_col = color_from_certainty(certainty)

    lines.append("")
    lines.append("Alt-Top Certainty (Composite)")
    lines.append(
        f"• Certainty: {cert_col} {certainty}/100 (yellow ≥ 40, red ≥ 70)")
    impacts = sorted([(label, raw, w, (raw/100.0)*w) for (label, raw, w) in subs],
                     key=lambda x: x[3], reverse=True)[:5]
    lines.append("• Top drivers:")
    for label, raw, w, _ in impacts:
        col = RED if raw >= 70 else (YELLOW if raw >= 40 else GREEN)
        lines.append(
            f"  • {label}: {col} {int(round(raw))}/100 (w={int(round(w*100))}%)")

    return "\n".join(lines), certainty

# -----------------------------
# Telegram bot commands
# -----------------------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Hey! I’ll keep an eye on cycle-top risk and send you updates.\n\n"
        "Commands:\n"
        "• /assess – Run a fresh snapshot (uses your profile).\n"
        "• /assess conservative|moderate|aggressive – Snapshot with a different risk profile.\n\n"
        "I also post a once-daily summary and check alerts every 15 minutes."
    )
    await update.message.reply_text(msg)


async def cmd_assess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    profile = os.environ.get("DEFAULT_PROFILE", "moderate").lower().strip()
    if profile not in ("conservative", "moderate", "aggressive"):
        profile = "moderate"
    if context.args:
        p = (context.args[0] or "").lower().strip()
        if p in ("conservative", "moderate", "aggressive"):
            profile = p
    await update.message.reply_text("Running assessment…")
    m = await build_metrics(profile)
    text, _ = build_text(m)
    await update.message.reply_text(text)

# -----------------------------
# Push jobs
# -----------------------------


async def push_summary(app: Application, chat_id: int, profile: str):
    try:
        m = await build_metrics(profile)
        text, _ = build_text(m)
        await app.bot.send_message(chat_id=chat_id, text=text)
    except Exception as e:
        log.exception("push_summary failed: %s", e)


async def push_alerts(app: Application, chat_id: int, profile: str):
    try:
        m = await build_metrics(profile)
        _, certainty = build_text(m)
        if certainty >= 70:
            await app.bot.send_message(chat_id=chat_id, text=f"⚠️ Alt-Top Certainty is {certainty}/100 (red). Consider caution.")
    except Exception as e:
        log.exception("push_alerts failed: %s", e)

# -----------------------------
# Health server (aiohttp) in background thread
# -----------------------------


def start_health_server():
    import threading
    import asyncio
    from aiohttp import web

    async def handle_health(request):
        return web.json_response({"ok": True, "time": datetime.now(timezone.utc).isoformat()})

    def runner():
        async def runner_coro():
            app = web.Application()
            app.router.add_get("/health", handle_health)
            runner = web.AppRunner(app)
            await runner.setup()
            port = int(os.environ.get("PORT", "8080"))
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()
            log.info("Health server listening on :%d", port)
            while True:
                await asyncio.sleep(3600)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner_coro())

    t = threading.Thread(target=runner, daemon=True)
    t.start()

# -----------------------------
# Main
# -----------------------------


def parse_chat_id() -> Optional[int]:
    v = os.environ.get("TARGET_CHAT_ID", "").strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        log.warning("TARGET_CHAT_ID is not a valid integer: %r", v)
        return None


if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN env var required")

    profile = os.environ.get("DEFAULT_PROFILE", "moderate").lower().strip()
    if profile not in ("conservative", "moderate", "aggressive"):
        profile = "moderate"

    start_health_server()

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("assess", cmd_assess))

    chat_id = parse_chat_id()
    if chat_id is not None:
        jq = app.job_queue
        hour = int(os.environ.get("SUMMARY_HOUR_UTC", "13"))
        minute = int(os.environ.get("SUMMARY_MINUTE_UTC", "0"))
        jq.run_daily(
            lambda ctx: asyncio.create_task(
                push_summary(app, chat_id, profile)),
            time=dtime(hour=hour, minute=minute, tzinfo=timezone.utc),
            name="daily_summary",
        )
        jq.run_repeating(
            lambda ctx: asyncio.create_task(
                push_alerts(app, chat_id, profile)),
            interval=900, first=60, name="alerts_15m"
        )
        log.info(
            "Scheduler started (JobQueue): daily summary at %02d:%02d UTC, alerts every 15m", hour, minute)
    else:
        log.info("Scheduler disabled: TARGET_CHAT_ID not set.")

    log.info("Bot running. Press Ctrl+C to exit.")
    # IMPORTANT: don't await; this manages its own loop and avoids 'loop already running'
    app.run_polling(allowed_updates=Update.ALL_TYPES)
