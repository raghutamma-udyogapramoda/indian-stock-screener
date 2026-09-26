"""
API Data Audit & Verification Generator.
Fetches real data from the APIs (Yahoo Finance, Breeze) for NSE Equities and MCX Commodities,
computes indicators, and writes a comprehensive, human-readable text file (API_DATA_AUDIT_REPORT.txt)
so that admins and auditors can cross-reference the data against NSE India, MCX India, TradingView, etc.
"""

from datetime import datetime
from pathlib import Path
import sys
import pandas as pd

# Windows Unicode console support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.indicators import enrich_with_indicators
from core.universe import UniverseManager
from providers.breeze_static_provider import BreezeStaticProvider
from providers.yfinance_provider import YahooFinanceProvider
from screeners.breakdown import BreakdownScreener
from screeners.breakout import BreakoutScreener
from screeners.btst import BTSTScreener
from screeners.swing import SwingScreener


def generate_audit_report(output_file: str = "API_DATA_AUDIT_REPORT.txt"):
    target_path = Path(output_file)
    print(f"🔍 Generating API verification report: {target_path}...")

    # Assets to audit across Equities and Commodities
    audit_symbols = [
        # Major NSE Equities
        {"symbol": "RELIANCE", "exchange": "NSE", "provider": "yfinance", "desc": "Nifty 50 Heavyweight (Energy / Telecom)"},
        {"symbol": "TCS", "exchange": "NSE", "provider": "yfinance", "desc": "IT Major (Nifty 50)"},
        {"symbol": "TRENT", "exchange": "NSE", "provider": "yfinance", "desc": "Retail Leader (Breakout Candidate)"},
        {"symbol": "ASIANPAINT", "exchange": "NSE", "provider": "yfinance", "desc": "Paints & Chemicals (Support Breakdown Candidate)"},
        
        # Major MCX Commodities (Live Yahoo Finance futures fallback + Breeze schema)
        {"symbol": "GOLD", "exchange": "MCX", "provider": "yfinance", "desc": "Gold Futures (GC=F / MCX Bullion)"},
        {"symbol": "CRUDEOIL", "exchange": "MCX", "provider": "yfinance", "desc": "Crude Oil Futures (CL=F / MCX Energy)"},
        {"symbol": "SILVER", "exchange": "MCX", "provider": "yfinance", "desc": "Silver Futures (SI=F / MCX Bullion)"},

        # Breeze API Static Candidate Audit
        {"symbol": "GOLD", "exchange": "MCX", "provider": "breeze_static", "desc": "Breeze API Format: Gold 52W Breakout"},
        {"symbol": "CRUDEOIL", "exchange": "MCX", "provider": "breeze_static", "desc": "Breeze API Format: Crude Oil Support Breakdown"},
    ]

    yf_provider = YahooFinanceProvider(cache_ttl_hours=4.0)
    breeze_provider = BreezeStaticProvider()

    bo_screener = BreakoutScreener()
    bd_screener = BreakdownScreener()
    btst_screener = BTSTScreener()
    swing_screener = SwingScreener()

    lines = []
    lines.append("=" * 90)
    lines.append("INDIAN EQUITIES & MCX COMMODITIES - API DATA AUDIT & VERIFICATION REPORT")
    lines.append(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
    lines.append("Purpose: Provide administrators & traders with exact raw API candle feeds,")
    lines.append("         computed indicator metrics, and screening qualifications for verification.")
    lines.append("=" * 90)
    lines.append("")
    lines.append("TABLE OF CONTENTS / SUMMARY")
    lines.append("-" * 90)
    lines.append(f"{'Symbol':<12} | {'Exchange':<8} | {'Provider':<14} | {'Latest Date':<12} | {'Close (₹)':<12} | {'20d Vol Avg':<12}")
    lines.append("-" * 90)

    cached_results = []

    for item in audit_symbols:
        sym = item["symbol"]
        prov_name = item["provider"]
        provider = yf_provider if prov_name == "yfinance" else breeze_provider

        try:
            df = provider.fetch_ohlcv(sym, period="1y", interval="1d" if prov_name == "yfinance" else "1day")
        except Exception as e:
            df = None

        if df is not None and not df.empty:
            df_enriched = enrich_with_indicators(df.copy())
            last = df_enriched.iloc[-1]
            last_date = str(df_enriched.index[-1])[:10]
            close_p = float(last["close"])
            vol_avg = float(last.get("volume_sma_20", 0))

            cached_results.append({
                "item": item,
                "df": df,
                "df_enriched": df_enriched,
                "last_date": last_date,
                "close_p": close_p,
                "vol_avg": vol_avg
            })
            lines.append(f"{sym:<12} | {item['exchange']:<8} | {prov_name:<14} | {last_date:<12} | ₹{close_p:<11.2f} | {vol_avg:<12.0f}")
        else:
            lines.append(f"{sym:<12} | {item['exchange']:<8} | {prov_name:<14} | FAILED TO FETCH DATA")

    lines.append("-" * 90)
    lines.append("")
    lines.append("=" * 90)
    lines.append("DETAILED ASSET AUDIT LOGS (CANDLES & INDICATOR PROOFS)")
    lines.append("=" * 90)

    for record in cached_results:
        item = record["item"]
        sym = item["symbol"]
        prov = item["provider"]
        df = record["df"]
        df_enriched = record["df_enriched"]
        last = df_enriched.iloc[-1]
        prev = df_enriched.iloc[-2] if len(df_enriched) > 1 else last

        clean_sym = UniverseManager.to_clean_symbol(sym)
        yf_ticker = UniverseManager.to_yfinance_symbol(sym)

        lines.append("")
        lines.append(f">>> ASSET: {sym} ({item['desc']})")
        lines.append(f"    Exchange: {item['exchange']} | Data Provider: {prov.upper()}")
        lines.append(f"    API Ticker Queried: {yf_ticker if prov == 'yfinance' else clean_sym}")
        lines.append(f"    Total Historical Daily Candles Loaded: {len(df)}")
        lines.append(f"    Data Date Span: {str(df.index[0])[:10]} to {str(df.index[-1])[:10]}")
        lines.append("-" * 90)

        # 1. Indicator Snapshot
        lines.append("    [TECHNICAL INDICATORS COMPUTED FOR LATEST CANDLE]")
        lines.append(f"    • Close Price: ₹{float(last['close']):.2f}")
        lines.append(f"    • Change from Prev Close: {((float(last['close']) - float(prev['close'])) / float(prev['close'])) * 100:+.2f}%")
        lines.append(f"    • Volume: {float(last['volume']):,.0f} (20-day Volume SMA: {float(last.get('volume_sma_20', 0)):,.0f})")
        lines.append(f"    • Relative Volume (RVOL): {float(last.get('rvol_20', 1.0)):.2f}x")
        lines.append(f"    • RSI (14-period Wilder's): {float(last.get('rsi_14', 50.0)):.2f}")
        lines.append(f"    • EMA 20: ₹{float(last.get('ema_20', 0)):.2f}")
        lines.append(f"    • EMA 50: ₹{float(last.get('ema_50', 0)):.2f}")
        lines.append(f"    • SMA 200: ₹{float(last.get('sma_200', 0)):.2f}")
        lines.append(f"    • Bollinger Upper: ₹{float(last.get('bb_upper', 0)):.2f} | Lower: ₹{float(last.get('bb_lower', 0)):.2f}")
        lines.append(f"    • Bollinger Bandwidth (BBW): {float(last.get('bb_bandwidth', 0)):.4f}")
        lines.append(f"    • 52-Week High: ₹{float(df['high'].tail(252).max()):.2f} (Dist: {((float(df['high'].tail(252).max()) - float(last['close'])) / float(df['high'].tail(252).max())) * 100:.2f}%)")
        lines.append(f"    • 52-Week Low: ₹{float(df['low'].tail(252).min()):.2f} (Dist: {((float(last['close']) - float(df['low'].tail(252).min())) / float(df['low'].tail(252).min())) * 100:.2f}%)")
        lines.append(f"    • 20-Day Swing Pivot High: ₹{float(df['high'].iloc[-21:-1].max() if len(df)>=21 else last['high']):.2f}")
        lines.append(f"    • 20-Day Swing Pivot Low:  ₹{float(df['low'].iloc[-21:-1].min() if len(df)>=21 else last['low']):.2f}")
        lines.append(f"    • Close Location Value (CLV): {float(last.get('clv', 0.5)):.3f} (0=Low, 1=High)")

        # 2. Screener Evaluations
        lines.append("")
        lines.append("    [ALGORITHMIC SCREENER EVALUATION]")
        bo_match = bo_screener.screen(sym, df)
        bd_match = bd_screener.screen(sym, df)
        btst_match = btst_screener.screen(sym, df)
        swing_match = swing_screener.screen(sym, df)

        lines.append(f"    • Breakout Screener: {'QUALIFIED (Score: ' + str(bo_match['score']) + ')' if bo_match else 'REJECTED'}")
        if bo_match:
            lines.append(f"      Reasons: {', '.join(bo_match['reasons'])}")
        lines.append(f"    • Breakdown Screener: {'QUALIFIED (Score: ' + str(bd_match['score']) + ')' if bd_match else 'REJECTED'}")
        if bd_match:
            lines.append(f"      Reasons: {', '.join(bd_match['reasons'])}")
        lines.append(f"    • BTST Screener: {'QUALIFIED (Score: ' + str(btst_match['score']) + ')' if btst_match else 'REJECTED'}")
        lines.append(f"    • Swing Screener: {'QUALIFIED (Score: ' + str(swing_match['score']) + ')' if swing_match else 'REJECTED'}")

        # 3. Raw Candlestick Printout (Last 10 trading sessions)
        lines.append("")
        lines.append("    [EXACT RAW CANDLESTICK DATA FROM API (LAST 10 SESSIONS)]")
        lines.append(f"    {'Date':<12} | {'Open (₹)':<10} | {'High (₹)':<10} | {'Low (₹)':<10} | {'Close (₹)':<10} | {'Volume':<12}")
        lines.append("    " + "-" * 72)
        recent_candles = df.tail(10)
        for dt, row in recent_candles.iterrows():
            d_str = str(dt)[:10]
            lines.append(f"    {d_str:<12} | {float(row['open']):<10.2f} | {float(row['high']):<10.2f} | {float(row['low']):<10.2f} | {float(row['close']):<10.2f} | {float(row['volume']):<12,.0f}")

        lines.append("=" * 90)

    # Cross-verification Guide
    lines.append("")
    lines.append("HOW ADMINS CAN CROSS-VERIFY THESE NUMBERS EXTERNALLY:")
    lines.append("-" * 90)
    lines.append("1. National Stock Exchange of India (NSE):")
    lines.append("   - Visit: https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE")
    lines.append("   - Verify: Compare the Open, High, Low, Close, and Total Traded Volume of the latest date.")
    lines.append("2. Multi Commodity Exchange of India (MCX):")
    lines.append("   - Visit: https://www.mcxindia.com/market-data/market-watch")
    lines.append("   - Verify: Compare Gold, Crude Oil, Silver contract settlement price & high/low range.")
    lines.append("3. Yahoo Finance / TradingView:")
    lines.append("   - Search: RELIANCE.NS, GC=F (Gold), CL=F (Crude Oil).")
    lines.append("   - Verify: Confirm that daily candlesticks and 52-week High/Low match.")
    lines.append("-" * 90)
    lines.append("END OF AUDIT REPORT")
    lines.append("=" * 90)

    report_content = "\n".join(lines)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"✅ Audit report successfully generated at: {target_path.resolve()}")
    return target_path


if __name__ == "__main__":
    generate_audit_report()
