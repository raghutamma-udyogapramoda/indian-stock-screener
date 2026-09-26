"""
Master CLI Entrypoint for Stock Screener and AI Advisor.
Usage:
    python main.py --universe NIFTY_50 --strategy all
    python main.py --universe NIFTY_500 --strategy breakout
    python main.py --universe NSE_FO --strategy options
"""

import argparse
import sys
import time
from typing import Dict, List
import pandas as pd

# Ensure proper Unicode/emoji handling on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai.advisor import AIAdvisor
from core.indices import IndexDerivativesAnalyzer
from core.universe import UniverseManager
from providers.breeze_provider import BreezeProvider
from providers.breeze_static_provider import BreezeStaticProvider
from providers.nse_direct_provider import NSEDirectProvider
from providers.yfinance_provider import YahooFinanceProvider
from screeners.breakdown import BreakdownScreener
from screeners.breakout import BreakoutScreener
from screeners.btst import BTSTScreener
from screeners.intraday import IntradayScreener
from screeners.options_fno import OptionsScreener
from screeners.swing import SwingScreener


def run_pipeline(
    universe_name: str = "NIFTY_50",
    strategy: str = "all",
    top_n: int = 5,
    enable_ai: bool = True,
    provider_name: str = "yfinance"
):
    print("=" * 75)
    print(f"🚀 INDIAN MARKET SCREENER & AI ADVISOR")
    print(f"   Universe: {universe_name} | Provider: {provider_name.upper()} | Strategy: {strategy.upper()} | Top Picks: {top_n}")
    print("=" * 75)

    # STEP 1: Load Universe Definition
    # Retrieves ticker list from NSE/MCX archives or curated seeds (NIFTY_50, NSE_FO, NIFTY_500, MCX).
    t0 = time.time()
    tickers = UniverseManager.get_tickers(universe_name)
    print(f"📋 Loaded {len(tickers)} symbols for universe '{universe_name}'")

    # STEP 2: Concurrently Ingest Market Data with Chosen Provider
    if provider_name == "nse_direct":
        provider = NSEDirectProvider(cache_ttl_hours=4.0)
        print("🏛️ Using Direct National Stock Exchange (NSE India) Feed (Official Bhavcopy).")
    elif provider_name == "breeze_static":
        provider = BreezeStaticProvider()
        print("📁 Using ICICI Direct Breeze Static Offline Fixtures.")
    elif provider_name == "breeze":
        provider = BreezeProvider()
        if not provider.is_authenticated:
            print("⚠️ Warning: Live Breeze credentials not authenticated. Falling back to Breeze Static Provider.")
            provider = BreezeStaticProvider()
        else:
            print("🔗 Connected to live ICICI Direct Breeze API session.")
    else:
        provider = YahooFinanceProvider(cache_ttl_hours=4.0)

    print(f"⏳ Syncing market data and indicators for {len(tickers)} assets...")
    data_dict = provider.fetch_batch_ohlcv(tickers, period="1y", interval="1d" if provider_name == "yfinance" else "1day", max_workers=15)
    fetch_time = time.time() - t0
    print(f"✅ Data ready for {len(data_dict)} assets in {fetch_time:.2f}s\n")

    # Check for Indian Market Indices
    index_data = {sym: df for sym, df in data_dict.items() if UniverseManager.is_index(sym)}
    equity_data = {sym: df for sym, df in data_dict.items() if not UniverseManager.is_index(sym)}

    if index_data:
        print("=" * 80)
        print("🏛️ INDIAN MARKET & SECTORAL INDICES DERIVATIVES HUB")
        print("=" * 80)
        index_analyses = IndexDerivativesAnalyzer.analyze_batch(index_data)
        if index_analyses:
            top_idx = index_analyses[0]
            print(f"🏆 TOP RECOMMENDED INDEX TODAY: {top_idx['display_name']} ({top_idx['signal_badge']}) | Score: {top_idx['profitability_score']}/100")
            print("-" * 80)

            # Two-Way Action Levels Table (Buy Above / Downside Possible Below)
            print("\n🎯 TWO-WAY ACTION LEVELS (BUY ABOVE / DOWNSIDE POSSIBLE BELOW):")
            action_rows = []
            for ia in index_analyses:
                al = ia.get("action_levels", {})
                action_rows.append({
                    "Index": ia["display_name"],
                    "LTP": f"₹{ia['close']:,.1f}",
                    "🟢 Buy Above": f"₹{al.get('buy_above', 0):,.1f}",
                    "Upside Targets": f"T1: ₹{al.get('upside_target_1', 0):,.1f} | T2: ₹{al.get('upside_target_2', 0):,.1f}",
                    "Upside SL": f"₹{al.get('upside_stop_loss', 0):,.1f}",
                    "🔴 Downside Below": f"₹{al.get('sell_below', 0):,.1f}",
                    "Downside Targets": f"T1: ₹{al.get('downside_target_1', 0):,.1f} | T2: ₹{al.get('downside_target_2', 0):,.1f}",
                    "Downside SL": f"₹{al.get('downside_stop_loss', 0):,.1f}",
                    "Chop Range": al.get('chop_zone', '')
                })
            print(pd.DataFrame(action_rows).to_string(index=False))

            print("\n📋 STRATEGIC PLAYBOOK & THESIS PER INDEX:")
            for ia in index_analyses:
                al = ia.get("action_levels", {})
                print(f"• {ia['display_name']} (LTP: ₹{ia['close']:,.1f}):\n   {al.get('trade_thesis', '')}\n   Option Plays -> Bullish: {al.get('upside_option_play')} | Bearish: {al.get('downside_option_play')}")

            # Futures Table
            print("\n📈 INDEX FUTURES TRADING PLANS (BUY / SELL / HOLD):")
            futures_rows = []
            for ia in index_analyses:
                fp = ia["futures_plan"]
                futures_rows.append({
                    "Index": ia["display_name"],
                    "Signal": ia["signal_badge"],
                    "LTP": f"₹{ia['close']:,.1f}",
                    "Lot Size": ia["lot_size"],
                    "Change": f"{ia['change_pct']:+.2f}%",
                    "Entry Zone": fp["entry_zone"],
                    "Stop Loss": f"₹{fp['stop_loss']:,.1f}",
                    "Target 1": f"₹{fp['target_1']:,.1f}",
                    "R:R": fp["risk_reward_ratio"],
                    "Max Profit/Lot": fp["max_profit_target_1"],
                    "Score": f"{ia['profitability_score']}/100"
                })
            print(pd.DataFrame(futures_rows).to_string(index=False))

            # Options Table
            print("\n🎯 INDEX OPTIONS PLAYS & PROFITABILITY (PER LOT):")
            options_rows = []
            for ia in index_analyses:
                for opt in ia["options_setup"]["strategies"]:
                    options_rows.append({
                        "Index": ia["display_name"],
                        "Strategy": opt["name"],
                        "Contract": opt["recommended_contract"],
                        "Target Pts": opt["target_points"],
                        "Est. P&L/Lot": opt["est_profit_lot"],
                        "Max Risk/Lot": opt["est_risk_lot"],
                        "Suitability": opt["suitability"]
                    })
            print(pd.DataFrame(options_rows).to_string(index=False))
            print("=" * 80 + "\n")

        if not equity_data:
            print(f"✨ Completed index derivatives analysis in {time.time() - t0:.2f}s total.")
            return

    # STEP 3: Initialize Active Quantitative Screeners
    active_screeners = []
    strat = strategy.lower()
    if strat in ["all", "breakout"]:
        active_screeners.append(BreakoutScreener())
    if strat in ["all", "breakdown"]:
        active_screeners.append(BreakdownScreener())
    if strat in ["all", "swing"]:
        active_screeners.append(SwingScreener())
    if strat in ["all", "btst"]:
        active_screeners.append(BTSTScreener())
    if strat in ["all", "intraday"]:
        active_screeners.append(IntradayScreener())
    if strat in ["all", "options"]:
        active_screeners.append(OptionsScreener())
    elif strat == "ce":
        active_screeners.append(OptionsScreener(option_target="CE"))
    elif strat == "pe":
        active_screeners.append(OptionsScreener(option_target="PE"))

    # STEP 4: Run Algorithmic Filtering (The Quantitative Shield)
    # Evaluates each stock's OHLCV against strict trend, volatility, and volume criteria.
    all_candidates: List[dict] = []
    target_stocks = equity_data if equity_data else data_dict
    for sc in active_screeners:
        print(f"🔍 Screening for {sc.name}...")
        candidates = sc.screen_batch(target_stocks, top_n=top_n, benchmark_data=index_data)
        print(f"   Found {len(candidates)} qualified setup(s)")
        all_candidates.extend(candidates)

    if not all_candidates:
        print("\n⚠️ No stocks met the strict screening criteria today. Markets may be consolidating.")
        return

    # Print Quantitative Shortlist Summary Table
    print("\n" + "=" * 90)
    print("📊 QUANTITATIVE SCREENING SHORTLIST")
    print("=" * 90)
    summary_rows = []
    for c in all_candidates:
        retest_val = c.get("retest_support_50pct", 0)
        retest_str = f"₹{retest_val:,.1f}" if retest_val > 0 else "—"
        sector_val = c.get("sector", "Broad")
        tailwind_str = f"{sector_val} (Tailwind)" if c.get("has_sector_tailwind") else sector_val

        summary_rows.append({
            "Symbol": c["symbol"],
            "Category": c["category"],
            "LTP (₹)": c["close"],
            "Change %": f"{c['change_pct']:+.2f}%",
            "50% Retest": retest_str,
            "Sector Tailwind": tailwind_str,
            "Panic Resilient": "YES" if c.get("panic_day_resilient") else "—",
            "Vol 2X": "YES" if c.get("vol_confirmed_2x") else "—",
            "RVol(9d)": f"{c.get('rvol_9', c['rvol']):.1f}x",
            "RSI": c["rsi"],
            "Score": c["score"]
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    # STEP 5: AI Trade Advisor Evaluation (The LLM Token Shield)
    # Serializes ONLY the top 5-15 shortlisted setups into a lightweight JSON feature vector (< 1.5k tokens)
    # and passes it to Google Gemini Flash (or mathematical precision fallback).
    if enable_ai:
        print("\n" + "=" * 75)
        print("🤖 AI TRADE ADVISOR (Trade Plans & Risk Levels)")
        print("=" * 75)
        advisor = AIAdvisor()
        status_msg = "Google Gemini Flash (Live AI)" if advisor.is_ai_ready else "Algorithmic Precision Engine"
        print(f"Generating plans using: {status_msg}...\n")

        plans = advisor.analyze_candidates(all_candidates)
        for p in plans:
            print(f"[{p.get('category')}] {p.get('symbol')} ── Action: {p.get('trade_action')}")
            print(f"   Entry: ₹{p.get('entry_price')} | Stop Loss: ₹{p.get('stop_loss')} | Target 1: ₹{p.get('target_1')} | Target 2: ₹{p.get('target_2')}")
            print(f"   R:R: {p.get('risk_reward_ratio')} | Timeframe: {p.get('timeframe')} | Conviction: {p.get('conviction')}")
            print(f"   Thesis: {p.get('thesis')}\n")

    print("=" * 75)
    print(f"✨ Completed analysis in {time.time() - t0:.2f}s total.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Stock Market Screener & AI Advisor")
    parser.add_argument("--universe", type=str, default="NIFTY_50", help="NIFTY_50, NIFTY_500, NSE_FO, INDICES, MCX, or custom comma-separated tickers")
    parser.add_argument(
        "--provider",
        type=str,
        default="yfinance",
        choices=["yfinance", "nse_direct", "breeze", "breeze_static"],
        help="Data provider: yfinance, nse_direct (official NSE Bhavcopy), breeze (live ICICI), or breeze_static"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "breakout", "breakdown", "swing", "btst", "intraday", "options", "ce", "pe"],
        help="Strategy to screen: all, breakout, breakdown, swing, btst, intraday, options, ce, pe"
    )
    parser.add_argument("--top", type=int, default=5, help="Number of top candidates per category")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI Trade Plan generation")

    args = parser.parse_args()
    run_pipeline(
        universe_name=args.universe,
        strategy=args.strategy,
        top_n=args.top,
        enable_ai=not args.no_ai,
        provider_name=args.provider
    )
