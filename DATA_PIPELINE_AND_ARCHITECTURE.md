# Indian Equities AI Screener & Advisor: Architecture & Data Flow

This document details:
1. **Where stock market data originates** (Providers, APIs, and caching)
2. **How data is computed, enriched, and quantitatively filtered** (Formulas and Screeners)
3. **What exact data is sent to the LLM** (Token Shield, feature vector schema, and LLM output)

---

## 1. System Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────┐
                        │                     DATA SOURCES                        │
                        │  1. Yahoo Finance (Daily OHLCV, 1yr, auto-adjusted)     │
                        │  2. NSE Official Archives (Nifty 50, Nifty 500 CSVs)    │
                        │  3. ICICI Direct Breeze API (Live ticks, NFO Option OI) │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │             CONCURRENCY & LOCAL CACHE LAYER             │
                        │  - ThreadPoolExecutor (15 parallel workers)             │
                        │  - Apache Parquet storage (data/cache/<SYMBOL>.parquet) │
                        │  - Sub-second warm-cache retrieval (< 0.5s for 200 stks)│
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │         INTERNAL VECTORIZED INDICATOR COMPUTATION       │
                        │  - Pure Pandas & NumPy (No C-library / TA-Lib needed)   │
                        │  - EMAs (9, 20, 50), SMA 200, RSI 14 (Wilder's Smooth)  │
                        │  - Bollinger Bands + Bandwidth (BBW Squeeze detection)  │
                        │  - ATR 14, Relative Volume (RVol 20d), CLV (Close Loc) │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │             QUANTITATIVE SCREENING SHIELD               │
                        │  Filters 200-500 stocks down to top 5-15 setups:       │
                        │   • Breakout: Stage-2 trend, 52W/20D high, BB squeeze  │
                        │   • Breakdown: Stage-4 trend, support breach, CLV <=0.35│
                        │   • Swing: Pullback to EMA20/50, lower BB, hammer wick │
                        │   • BTST: Top 25% close (CLV >= 0.70), vol surge       │
                        │   • Intraday: ATR expansion (>= 50%), EMA 9/20 trend   │
                        │   • Options: CE (Call) & PE (Put), ATM/ITM/OTM strikes │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │                   LLM TOKEN SHIELD                      │
                        │  Serializes ONLY top setups into ultra-compact JSON     │
                        │  - Discards raw candlestick arrays (avoids 100k+ tokens)│
                        │  - Kept under 1,500 tokens (~$0.0004 on Gemini Flash)   │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │                LLM TRADE ADVISOR / ENGINE               │
                        │  Primary: Google Gemini 2.0 / 1.5 Flash (Structured)   │
                        │  Fallback: Built-in Algorithmic Mathematical Engine     │
                        │  Generates: Entry, Stop-Loss, T1, T2, R:R, Thesis       │
                        └───────────────────────────┬─────────────────────────────┘
                                                    │
                                        ┌───────────┴───────────┐
                                        ▼                       ▼
                        ┌────────────────────────┐  ┌────────────────────────┐
                        │     CLI INTERFACE      │  │  STREAMLIT DASHBOARD   │
                        │      (main.py)         │  │      (ui/app.py)       │
                        └────────────────────────┘  └────────────────────────┘
```

---

## 2. Where Stock Data Comes From

The application uses a pluggable architecture defined in `providers/base.py` (`BaseDataProvider`). Data is fetched from three tiers:

### Tier 1: Direct Official Exchange Ingestion (`providers/nse_direct_provider.py`)
- **Direct Access without Brokers or Yahoo**: Downloads official daily trade Bhavcopies directly from NSE India archives:
  - Daily Full Bhavcopy: `https://archives.nseindia.com/products/content/sec_bhavdata_full_<DDMMYYYY>.csv`
  - Ingests all **3,500+ listed NSE equities in a single 390 KB CSV download**, eliminating hundreds of individual API calls.
- **Exclusive Exchange Metric — Delivery Percentage (`DELIV_PER`)**:
  - Yahoo Finance and standard broker basic feeds only give total traded volume.
  - NSE Direct provides exact **Deliverable Quantity (`DELIV_QTY`)** and **Delivery Percentage (`DELIV_PER`)**. High delivery percentage (e.g. $> 60\%$) confirms institutional accumulation or true breakout buying versus speculative intraday noise.
- **Auto-Fallback & Lookback**: Automatically probes the current day and past trading days if markets are closed or Bhavcopy processing is underway. Merges with historical cached bars to compute rolling EMAs, 52-week highs, and RSI.

### Tier 2: Yahoo Finance (`providers/yfinance_provider.py`)
- **What it provides**: 1 year of daily historical OHLCV candles (`open`, `high`, `low`, `close`, `volume`).
- **Ticker Conversion**: Converts Indian symbols into Yahoo Finance tickers using `core/universe.py` (`UniverseManager.to_yfinance_symbol`):
  - NSE stocks: `RELIANCE` $\rightarrow$ `RELIANCE.NS`
  - BSE stocks: `RELIANCE` $\rightarrow$ `RELIANCE.BO`
- **Parallel Multi-threading**: Uses Python's `concurrent.futures.ThreadPoolExecutor(max_workers=15)` to download 50 to 200 stocks concurrently in ~10–25 seconds on cold run.
- **Fast Local Parquet Caching (`core/cache.py`)**:
  - Saved as columnar Apache Parquet files (`data/cache/<SYMBOL>.parquet`) using `pyarrow`.
  - Configurable Time-To-Live (TTL, default 4 hours).
  - On warm cache, loading 184 stocks takes **under 0.5 seconds** with **zero external network requests**.

### Tier 3: Official National Stock Exchange (NSE) Archives (`core/universe.py`)
- **What it provides**: Official constituent index lists directly from NSE servers:
  - Nifty 50: `https://archives.nseindia.com/content/indices/ind_nifty50list.csv`
  - Nifty 500: `https://archives.nseindia.com/content/indices/ind_nifty500list.csv`
- **Liquid F&O Universe (`NSE_FO`)**: A built-in curated seed list of 181 active, high-volume derivatives underlying stocks on the NSE.

### Tier 4: ICICI Direct Breeze API (`providers/breeze_provider.py` & `providers/breeze_static_provider.py`)
- **What it provides**: Institutional data for active ICICI Direct account holders and offline static fixtures:
  - Supports both **NSE Equities** (`exchange_code="NSE"`, `product_type="cash"`) and **MCX Commodities** (`exchange_code="MCX"`, `product_type="commodity"`).
  - Historical OHLCV via `client.get_historical_data_v2()`.
  - Option Chain quotes (`NFO`) with Open Interest (OI) buildup dynamics (Long Buildup, Short Covering, Short Buildup, Long Unwinding).
  - **Static Offline Mode (`BreezeStaticProvider`)**: Reads authentic Breeze JSON response payloads from `data/static_breeze/<SYMBOL>.json` for testing Breakout and Breakdown candidate detection without an active daily session token.
- **Configuration**:
  ```ini
  BREEZE_API_KEY=your_key
  BREEZE_SECRET_KEY=your_secret
  BREEZE_SESSION_TOKEN=your_token
  ```

### Tier 5: MCX Commodities Universe (`core/universe.py`)
- **Supported Commodities**: `GOLD`, `SILVER`, `CRUDEOIL`, `NATURALGAS`, `COPPER`, `ZINC`, `ALUMINIUM`.
- **Exchange Detection**: `UniverseManager.get_exchange(symbol)` routes queries to `exchange_code="MCX"`.
- **Yahoo Finance Fallback**: Mapped to international commodity futures (`GOLD` $\rightarrow$ `GC=F`, `CRUDEOIL` $\rightarrow$ `CL=F`, `SILVER` $\rightarrow$ `SI=F`, `NATURALGAS` $\rightarrow$ `NG=F`, `COPPER` $\rightarrow$ `HG=F`).
- **Contract Lot Volume Calibration**: Screeners automatically adjust volume thresholds for commodities (lot sizes $\ge 50$ contracts vs equity shares $\ge 20,000$).

---

## 3. How Data is Filtered Internally

Filtering occurs in two sequential computational phases before any data touches the LLM:

### Phase A: Vectorized Mathematical Enrichment (`core/indicators.py`)
Each stock DataFrame is enriched with technical metrics in one vectorized pass using NumPy and Pandas:

| Indicator | Implementation / Formula | Trading Purpose |
|---|---|---|
| **EMA 9, 20, 50** | Exponential Moving Averages (`Series.ewm(span=N)`) | Trend direction and dynamic support/resistance |
| **SMA 200** | 200-period Simple Moving Average | Macro bull/bear regime filter |
| **RSI 14** | Wilder's smoothed Relative Strength Index | Momentum thrust and overbought/oversold reset |
| **Bollinger Bands** | 20 SMA $\pm 2\sigma$; $\text{Bandwidth} = \frac{\text{Upper} - \text{Lower}}{\text{Middle}}$ | Volatility contraction (squeeze) and expansion |
| **ATR 14** | Wilder's smoothed Average True Range | Volatility-adjusted stop-loss and target sizing |
| **RVol (20d)** | $\text{RVol} = \frac{\text{Current Volume}}{\text{SMA}_{20}(\text{Volume})}$ | Institutional accumulation vs distribution |
| **CLV** | $\text{CLV} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}}$ | Intraday buying pressure: $\ge 0.70$ (buyers won), $\le 0.30$ (sellers won) |

### Phase B: Strategy Screening Engines (`screeners/`)
The enriched stocks pass through algorithmic rule screeners:

1. **Breakout Screener (`screeners/breakout.py`)**:
   - **Conditions**: Close within 4% of 52-Week High OR breaking above 20-day swing high.
   - **Trend**: Stage-2 uptrend ($\text{Price} \ge \text{EMA}_{20} \ge \text{EMA}_{50}$).
   - **Volatility**: Bollinger Bandwidth within 35% of 40-day minimum (volatility squeeze).
   - **Confirmation**: Volume surge ($\text{RVol} \ge 1.25\times$), $\text{CLV} \ge 0.50$.

2. **Breakdown Screener (`screeners/breakdown.py`)**:
   - **Conditions**: Violating 20-day swing support or testing 52-Week Low.
   - **Trend**: Stage-4 downtrend ($\text{Price} \le \text{EMA}_{20} \le \text{EMA}_{50}$).
   - **Selling Pressure**: Heavy distribution volume ($\text{RVol} \ge 1.25\times$) with $\text{CLV} \le 0.35$ (sellers dominating close) and $\text{RSI} \le 48$.

3. **Swing Trading Screener (`screeners/swing.py`)**:
   - **Structure**: Medium-term uptrend ($\text{EMA}_{20} \ge \text{EMA}_{50}$ or $\text{Price} \ge \text{SMA}_{200}$).
   - **Value Pullback**: Price testing dynamic support (within 2.8% of EMA 20 / EMA 50 or bouncing off lower Bollinger Band).
   - **Reversal Trigger**: Bullish candle, hammer absorption wick, or $\text{CLV} \ge 0.55$.
   - **RSI Continuation Zone**: $38 \le \text{RSI} \le 64$.
   - **Asymmetric R:R**: Stop-loss at recent swing low, target at swing high (minimum $1:2.5$ R:R).

4. **BTST (Buy Today, Sell Tomorrow) Screener (`screeners/btst.py`)**:
   - **Accumulation**: Close in top 30% of day's range ($\text{CLV} \ge 0.70$) on a green session.
   - **Surge**: Relative Volume $\ge 1.25\times$.
   - **Momentum**: Testing or breaking 5-day swing high; $52 \le \text{RSI} \le 76$.

5. **Intraday Momentum Screener (`screeners/intraday.py`)**:
   - **Range Expansion**: Daily range $\ge 50\%$ of 14-day ATR.
   - **Trend Momentum**: $\text{Price} \ge \text{EMA}_9 \ge \text{EMA}_{20}$ with $\text{CLV} \ge 0.68$ (Bullish) OR $\text{Price} \le \text{EMA}_9 \le \text{EMA}_{20}$ with $\text{CLV} \le 0.32$ (Bearish).

6. **Options CE/PE Screener (`screeners/options_fno.py`)**:
   - **Call (CE) Setup**: Price thrust $\ge +0.7\%$, $\text{RVol} \ge 1.15\times$, $\text{RSI} \ge 54$, $\text{CLV} \ge 0.65$.
   - **Put (PE) Setup**: Price drop $\le -0.7\%$, $\text{RVol} \ge 1.15\times$, $\text{RSI} \le 46$, $\text{CLV} \le 0.35$.
   - **Calculations**: Auto-computes ATM strike, 1-strike ITM (high Delta), and 1-strike OTM (gamma leverage) using exchange strike step rules.

---

## 4. What Data is Sent to the LLM (The Token Shield)

### Why Raw Data is NOT Sent
A full historical dataset of 180 stocks over 1 year contains over **90,000 candle rows** (~450,000 numbers). Sending raw data to an LLM would cost 100,000+ tokens ($0.20–$0.50 per scan), hit rate limits, and suffer from LLM mathematical hallucination.

### The Compact Feature Vector (`ai/prompt_builder.py`)
Only the **top shortlisted candidates** (e.g. 5 to 15 stocks) that already passed the quantitative criteria are serialized into a lightweight JSON vector.

#### Example Payload Sent to Gemini:
```json
[
  {
    "symbol": "SUNTV",
    "category": "BREAKOUT",
    "ltp": 512.70,
    "change_pct": 4.59,
    "rvol": 3.62,
    "rsi_14": 66.9,
    "triggers": [
      "Multi-week resistance breakout (cleared 20-day high ₹495.2)",
      "Stage-2 Trend alignment (Price > EMA20 >= EMA50)",
      "Volume surge (3.6x 20-day avg)"
    ],
    "key_levels": {
      "breakout_pivot": 495.20,
      "dist_from_52w_high_pct": 12.4,
      "ema_20": 473.73,
      "ema_50": 468.10,
      "bb_bandwidth": 0.084
    }
  },
  {
    "symbol": "AUBANK",
    "category": "OPTIONS",
    "ltp": 1000.10,
    "change_pct": -5.02,
    "rvol": 3.25,
    "rsi_14": 35.8,
    "triggers": [
      "Option Play: Buy 1000 PE (ATM)",
      "Setup: MOMENTUM_EXPANSION (BEARISH)",
      "Price Change: -5.02% with 3.2x volume",
      "RSI: 35.8, ATR: 24.12"
    ],
    "key_levels": {
      "atm_strike": 1000.0,
      "itm_strike": 1025.0,
      "otm_strike": 975.0,
      "atr": 24.12,
      "clv": 0.08
    }
  }
]
```

#### Token Shield Economics:
- **Payload Size**: ~800 to 1,500 tokens total.
- **Cost**: Less than **$0.0004 per scan** using Gemini 2.0 Flash or 1.5 Flash.
- **Latency**: ~1.5 to 2.5 seconds.

---

## 5. What the LLM Returns

The LLM receives the system prompt with strict schema enforcement (`response_mime_type="application/json"`). It outputs trade plans with mathematically grounded risk levels:

```json
[
  {
    "symbol": "SUNTV",
    "category": "BREAKOUT",
    "trade_action": "BUY",
    "entry_price": 513.73,
    "stop_loss": 473.73,
    "target_1": 593.73,
    "target_2": 653.73,
    "risk_reward_ratio": "1:2.0",
    "timeframe": "2-4 weeks (Positional Breakout)",
    "conviction": "HIGH",
    "thesis": "High-volume multi-week breakout with 3.6x volume surge confirming institutional expansion. Stop loss placed below EMA 20 support."
  },
  {
    "symbol": "AUBANK",
    "category": "OPTIONS",
    "trade_action": "BUY_PE",
    "entry_price": 1000.10,
    "stop_loss": 1036.20,
    "target_1": 951.96,
    "target_2": 915.86,
    "risk_reward_ratio": "1:1.3",
    "timeframe": "Weekly / Monthly Expiry",
    "conviction": "HIGH",
    "thesis": "Aggressive institutional distribution with 3.2x volume breakdown triggering high-delta 1000 PE put option momentum."
  }
]
```

### Algorithmic Precision Fallback (`ai/advisor.py`)
If no Gemini API key is configured in `.env`, or if an API call fails or quota is exhausted, the application **automatically falls back to its built-in Mathematical Engine**:
- Evaluates ATR, support/resistance pivots, and directional CLV.
- Outputs identical format trade cards seamlessly so the user's trading workflow is never interrupted.

---

## 6. Indian Market & Sectoral Indices Derivatives Hub (`core/indices.py`)

A dedicated quantitative analysis engine for India's major market benchmarks and sectoral indices:
- **Covered Benchmarks**:
  - **NIFTY 50** (`^NSEI`): Large-cap benchmark (Lot size: 25, Step: 50)
  - **BANK NIFTY** (`^NSEBANK`): High-beta banking index (Lot size: 15, Step: 100)
  - **BSE SENSEX** (`^BSESN`): 30 Bluechip BSE benchmark (Lot size: 10, Step: 100)
  - **FIN NIFTY** (`NIFTY_FIN_SERVICE.NS`): Financial services (Lot size: 25, Step: 50)
  - **MIDCAP NIFTY** (`^NSEMDCP50`): Mid-cap momentum leaders (Lot size: 50, Step: 25)
  - **NIFTY IT** (`^CNXIT`): Software exporters & tech leaders (Lot size: 25, Step: 100)
  - **INDIA VIX** (`^INDIAVIX`): 30-day forward volatility expectation gauge

### 1. Futures Buy / Sell / Hold Signal Engine
- **Moving Average Alignment**: Evaluates Close vs EMA 20, EMA 50, and 200 SMA.
- **Momentum Confluence**: Combines RSI 14, **MACD Level-II** (MACD > Signal > 0), and **Aroon Oscillator (25)**.
- **Volatility Squeeze**: Tracks Bollinger Bandwidth (BBW) to detect pre-breakout volatility coils.
- **Signals**:
  - `🟢 BUY (LONG FUTURES)`: Stage-2 uptrend, RSI 52–68, positive MACD Level-II, Aroon Osc > +30.
  - `🔴 SELL (SHORT FUTURES)`: Stage-4 downtrend, RSI 32–48, negative MACD, Aroon Osc < -30.
  - `🟡 HOLD / NEUTRAL`: Choppy range-bound action inside Bollinger Bands.
- **Risk Architecture**: Computes exact Entry Range, Stop-Loss, Target 1, Target 2, R:R Ratio, and Max Rupee Profit per lot based on 14-period ATR index points.

### 2. Options Strategy Explorer & Profitability Ranking
- Automatically calculates ATM, ITM, and OTM strike steps.
- **Directional Strategies**: ATM Call (CE) or ATM Put (PE) buying with delta-calibrated target and stop-loss points.
- **Hedged Spreads**: Bull Call Spreads (Buy ATM + Sell OTM) or Bear Put Spreads (Buy ATM + Sell OTM) to eliminate theta decay and define risk.
- **Non-Directional Plays**: Long Straddle / Strangle during Bollinger squeezes, and Iron Condors during sideways regimes.
- **Profitability Ranking**: Ranks all indices from highest profit potential to lowest, highlighting the **#1 Best Index to Trade Today**.
