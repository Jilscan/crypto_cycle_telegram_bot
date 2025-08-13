
# Crypto Cycle Watch — Telegram Bot

A lightweight Telegram bot that tracks crypto-cycle/altseason top-risk signals and alerts you when multiple metrics get extreme.

## What it tracks (out of the box, free sources)
- **BTC dominance** (CoinGecko)
- **ETH/BTC price** (Binance)
- **Altcap/BTC ratio** (derived: (Total mcap - BTC - ETH) / BTC, via CoinGecko global)
- **Perp funding rates** for a few majors (Binance: BTC, ETH, SOL, XRP, DOGE)
- **Open interest** snapshots for BTC and ETH (Binance)
- **Google Trends** pulse for `crypto`, `bitcoin`, `ethereum` (requires `pytrends`)

## Optional (if you have API keys)
- **BTC MVRV Z-Score** (Glassnode)
- **BTC Exchange Netflows** (Glassnode)
- (You can also add other exchanges/bybit endpoints similarly.)

> Note: Many “on-chain” metrics (MVRV, net inflows) are paywalled by data vendors. This bot integrates with free/low-friction endpoints by default and uses Glassnode optionally.

---

## Quick start

1. **Create a Telegram Bot**
   - Talk to [@BotFather](https://t.me/BotFather) and create a bot. Copy the token.

2. **Local run (Python 3.10+)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp config.example.yml config.yml
   # Edit config.yml with your Telegram token and preferences
   python main.py
   ```

3. **Docker**
   ```bash
   docker build -t crypto-cycle-bot .
   docker run -e TZ=America/New_York --rm -v $(pwd)/config.yml:/app/config.yml:ro crypto-cycle-bot
   ```

---

## Commands

- `/start` — Register/chat intro.
- `/status` — Pulls a fresh snapshot of all metrics.
- `/settime HH:MM` — (Optional) Set your local daily summary time (24h).
- `/subscribe` — Subscribe to alerts/summaries in this chat.
- `/unsubscribe` — Stop alerts/summaries.

The bot also sends **proactive alerts** when multiple indicators exceed configured thresholds.

---

## Configuration (`config.yml`)

```yaml
telegram:
  token: "YOUR_TELEGRAM_BOT_TOKEN"

schedule:
  # Cron-like, but we use a fixed daily time for summaries; alerts run every 15 minutes.
  daily_summary_time: "13:00"  # 24h local time of the server/container

symbols:
  funding: ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
  oi: ["BTCUSDT", "ETHUSDT"]

thresholds:
  # If >= N of these are triggered in the same run, send a "Top Risk" alert
  min_flags_for_alert: 3

  btc_dominance_max: 60.0        # % — if dominance spikes, alts usually at risk
  eth_btc_ratio_max: 0.09        # e.g., ETHBTC > 0.09 historically near altseason euphoria (adjust to taste)
  altcap_btc_ratio_max: 1.8      # Alt market cap (ex BTC/ETH) vs BTC
  funding_rate_abs_max: 0.10     # 0.10% 8h funding; extreme if sustained
  open_interest_btc_usdt_usd_max: 20_000_000_000   # 20B USD (approximate, adjust)
  open_interest_eth_usdt_usd_max: 8_000_000_000    # 8B USD (approximate, adjust)
  google_trends_7d_avg_min: 75   # avg of 'crypto','bitcoin','ethereum'

glassnode:
  api_key: ""     # optional
  enable: false

# Optional chat to force alerts to (without /subscribe). Usually leave empty.
force_chat_id: ""

logging:
  level: "INFO"
```

> **Tuning matters.** Everyone’s thresholds differ. Start conservative, watch a cycle, and tighten once you see what “extreme” looks like in practice.

---

## How alerting logic works

Each run, the bot builds a **flag count** from these rules (you can change them in `main.py`):

- BTC dominance > `btc_dominance_max` → 1 flag
- ETH/BTC > `eth_btc_ratio_max` → 1 flag
- Altcap/BTC > `altcap_btc_ratio_max` → 1 flag
- Any tracked symbol funding |funding_rate| > `funding_rate_abs_max` → 1 flag
- BTC or ETH USD notional OI > thresholds → 1 flag (per coin)
- Google Trends 7d average across keywords ≥ `google_trends_7d_avg_min` → 1 flag
- (Optional) MVRV Z-Score above a high watermark → 1 flag
- (Optional) Positive exchange netflows spike → 1 flag

If the **total flags ≥ `min_flags_for_alert`**, the bot posts a **Top Risk Alert**.

---

## Extend it

- Add more data vendors or exchanges.
- Add ETH/BTC breadth (e.g., % of top-100 alts at 30D highs vs BTC).
- Wire in notifications for *cooling* signals too (e.g., sustained negative funding, sharp OI reset).

---

## Troubleshooting

- If the bot does not respond, confirm the token and that the process can reach Telegram.
- Some Binance futures endpoints can rate-limit. The code backs off, but you may need to lower frequency.
- Google Trends can throttle aggressive polling; we only fetch a 7D window as needed.



---

## Risk profiles (conservative • moderate • aggressive)

You can now choose how sensitive alerts should be.

**Set it in chat:**  
`/risk conservative` • `/risk moderate` • `/risk aggressive`  
Check current setting: `/getrisk`

**How it works**
- Each profile has its own thresholds and `min_flags_for_alert` in `config.yml` under `risk_profiles`.
- The bot remembers the profile *per chat* (stored in `profiles.yml` at runtime).
- Default profile is `moderate` (configurable via `default_profile`).

