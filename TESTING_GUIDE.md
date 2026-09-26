# Comprehensive Testing Guide: Breeze API & MCX Commodities

This guide explains how to test the **Trading Application**, with a specific focus on **ICICI Direct Breeze API static data testing**, **MCX commodities screening**, and **Breakout / Breakdown candidate detection**.

---

## 1. Quick Start: 3 Ways to Test

### Method 1: Automated Test Suite (Recommended)
Run the dedicated Breeze & MCX test suite. It tests both **MCX Commodities** and **NSE Equities** against Breakout and Breakdown screeners, and generates full trade plans:

```powershell
.\.venv\Scripts\python.exe test_breeze_mcx.py
```

**Expected Result:**
```
================================================================================
🧪 ICICI DIRECT BREEZE API & MCX COMMODITIES SCREENING TEST
================================================================================
📁 Mode: STATIC / OFFLINE BREEZE FIXTURES
📋 Ingesting data for 6 assets across NSE & MCX:
   • GOLD         | Exchange: MCX  | Type: MCX Commodity
   • TRENT        | Exchange: NSE  | Type: NSE Equity
   • CRUDEOIL     | Exchange: MCX  | Type: MCX Commodity
   • ASIANPAINT   | Exchange: NSE  | Type: NSE Equity
   • RELIANCE     | Exchange: NSE  | Type: NSE Equity
   • SILVER       | Exchange: MCX  | Type: MCX Commodity

🚀 BREAKOUT CANDIDATES DETECTED: 2 (GOLD, TRENT)
🔻 BREAKDOWN CANDIDATES DETECTED: 2 (CRUDEOIL, ASIANPAINT)

🎯 VERIFICATION CHECKLIST:
  [PASS] MCX Commodity Breakout detected (GOLD)
  [PASS] NSE Equity Breakout detected (TRENT)
  [PASS] MCX Commodity Breakdown detected (CRUDEOIL)
  [PASS] NSE Equity Breakdown detected (ASIANPAINT)
  [PASS] Control symbols filtered out (RELIANCE, SILVER - zero false positives)
🎉 ALL TESTS PASSED!
```

---

### Method 2: Command-Line Interface (CLI)

Run live or static scans across different universes and providers using `main.py`:

```powershell
# 1. Screen MCX Commodities using Breeze Static Data
.\.venv\Scripts\python.exe main.py --provider breeze_static --universe MCX --strategy all

# 2. Screen only Breakout candidates on MCX
.\.venv\Scripts\python.exe main.py --provider breeze_static --universe MCX --strategy breakout

# 3. Screen only Breakdown candidates on MCX
.\.venv\Scripts\python.exe main.py --provider breeze_static --universe MCX --strategy breakdown

# 4. Screen Nifty 50 equities using Yahoo Finance
.\.venv\Scripts\python.exe main.py --universe NIFTY_50 --strategy breakout

# 5. Screen liquid F&O universe (181 stocks)
.\.venv\Scripts\python.exe main.py --universe NSE_FO --strategy all
```

---

### Method 3: Interactive Web Dashboard (Streamlit)

Launch the web dashboard to visualize candlestick charts, technical indicators, and AI trade plans:

```powershell
.\.venv\Scripts\streamlit.exe run ui/app.py
```
Open your browser at **`http://localhost:8501`**.

In the sidebar:
1. **Data Feed Provider**: Choose `ICICI Breeze (Static / Test)` or `Yahoo Finance (Default)`.
2. **Select Asset Universe**: Choose `MCX Commodities`, `NIFTY_50`, `NSE_FO`, or `Custom`.
3. **Strategy Filter**: Choose `Breakout (Multi-Week & 52W)` or `Breakdown (Short & Distribution)`.
4. Click **`🔍 Run Live Market Screen`**.

---

## 2. ICICI Direct Breeze API Integration

### How Breeze API Data is Formatted
The ICICI Direct Breeze API (`breeze-connect` SDK) fetches historical candles via:
```python
breeze.get_historical_data_v2(
    interval="1day",
    from_date="2025-01-01T07:00:00.000Z",
    to_date="2026-01-01T07:00:00.000Z",
    stock_code="GOLD",      # or "RELIANCE"
    exchange_code="MCX",    # "NSE" for equities, "MCX" for commodities
    product_type="commodity"# "cash" for equities, "commodity" for MCX
)
```

The raw API JSON response follows this structure:
```json
{
  "Success": [
    {
      "datetime": "2026-09-25 09:15:00",
      "open": 80644.25,
      "high": 81622.25,
      "low": 80399.75,
      "close": 81500.00,
      "volume": 17600
    }
  ],
  "Status": 200,
  "Error": null
}
```

### Static Data Testing Pipeline
Static JSON response files are stored in:
```
data/static_breeze/
├── GOLD.json          # MCX Commodity Breakout Candidate (52W High surge)
├── TRENT.json         # NSE Equity Breakout Candidate
├── CRUDEOIL.json      # MCX Commodity Breakdown Candidate (Support floor collapse)
├── ASIANPAINT.json    # NSE Equity Breakdown Candidate
├── RELIANCE.json      # NSE Equity Neutral / Control (Oscillating channel)
└── SILVER.json        # MCX Commodity Neutral / Control
```

When `BreezeStaticProvider` is used, the system:
1. Ingests the exact Breeze API JSON fixture from `data/static_breeze/<SYMBOL>.json`.
2. Passes it through `BreezeProvider.parse_breeze_response()`.
3. Standardizes column names (`open`, `high`, `low`, `close`, `volume`) and index (`datetime`).
4. Computes 200 SMA, 50 EMA, 20 EMA, ATR, RSI, Bollinger Bands, RVOL, and CLV.
5. Executes the screeners without requiring network calls or broker tokens.

---

## 3. MCX Commodities Architecture

### Supported Commodities
The following major MCX commodity contracts are natively supported:
| Symbol | Commodity | Primary Exchange | Fallback Ticker (yfinance) | Lot Size Considerations |
| :--- | :--- | :--- | :--- | :--- |
| **`GOLD`** | Gold (10g / 1kg) | `MCX` | `GC=F` | Traded in contracts; adjusted volume threshold |
| **`SILVER`** | Silver (1kg) | `MCX` | `SI=F` | High beta commodity; volatility bands adjusted |
| **`CRUDEOIL`**| Crude Oil (100 bbl)| `MCX` | `CL=F` | Highly trend-driven; ideal for breakdown scans |
| **`NATURALGAS`**| Natural Gas | `MCX` | `NG=F` | Fast moving momentum setups |
| **`COPPER`** | Copper (2.5 MT) | `MCX` | `HG=F` | Industrial metal; economic cycle indicator |
| **`ZINC`** | Zinc (5 MT) | `MCX` | `ZNC=F` | Base metal swing candidate |
| **`ALUMINIUM`**| Aluminium (5 MT) | `MCX` | `ALI=F` | Base metal range trading |

### Domain-Specific Threshold Adaptation
In Indian equities, daily trading volumes are measured in individual shares (hundreds of thousands). In MCX commodities, volume is measured in **lots** (often 1,000 to 20,000 contracts).

The screeners automatically detect if an asset is an MCX commodity using `UniverseManager.is_commodity(symbol)`:
- **Equities Volume Filter**: `volume >= 20,000` (Breakout) / `volume >= 25,000` (Breakdown)
- **MCX Commodities Volume Filter**: `volume >= 50` contracts

This eliminates false negative rejections on liquid MCX contracts.

---

## 4. Breakout & Breakdown Screening Criteria

### Breakout Screener (`screeners/breakout.py`)
Screens for assets in an aggressive Stage-2 uptrend breaking through multi-week or 52-week resistance:
1. **Stage-2 Uptrend**: Price $\ge$ EMA20 and EMA20 $\ge$ EMA50 $\times$ 0.98.
2. **Resistance Cleared**: Price > 20-day swing high OR within 4% of 52-week high.
3. **Volatility Contraction**: Bollinger Bandwidth in lowest 35% of recent 40-day range (squeeze before expansion).
4. **Institutional Conviction**:
   - RVOL $\ge$ 1.25x 20-day average.
   - Close Location Value (CLV) $\ge$ 0.50 (closed in the upper half of day's range).

### Breakdown Screener (`screeners/breakdown.py`)
Screens for assets in a Stage-4 structural decline breaking support floors (for short-selling, Intraday shorts, or PE buying):
1. **Stage-4 Downtrend**: Price $\le$ EMA20 and EMA20 $\le$ EMA50.
2. **Support Violated**: Price breached prior 20-day swing low OR within 3.5% of 52-week low.
3. **Heavy Selling Pressure**: Close Location Value (CLV) $\le$ 0.35 (closed near the dead low of the session).
4. **Bearish Momentum**: RSI-14 $\le$ 48.0 (acceleration down).

---

## 5. Live Breeze Account Testing (Optional)

If you have an active ICICI Direct trading account:

1. Obtain your **App Key** and **Secret Key** from the ICICI Direct API portal.
2. Generate your daily **Session Token** by logging into ICICI Direct.
3. Add them to your `.env` file:
   ```env
   BREEZE_API_KEY=your_app_key_here
   BREEZE_SECRET_KEY=your_secret_key_here
   BREEZE_SESSION_TOKEN=your_daily_session_token_here
   ```
4. Run live testing with snapshot export:
   ```powershell
   .\.venv\Scripts\python.exe test_breeze_mcx.py --live
   ```
   This will query the live Breeze API and automatically save the latest raw JSON response payloads into `data/static_breeze/` for subsequent offline testing!
