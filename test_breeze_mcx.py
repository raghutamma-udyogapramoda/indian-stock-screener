"""
Automated Test Runner for ICICI Direct Breeze API & MCX Commodities Screening.
Verifies Breakout and Breakdown candidate detection across NSE Equities and MCX Commodities
using static Breeze API JSON fixtures (and optional live Breeze connection).

Usage:
    # 1. Run offline test using static Breeze API fixtures (NSE Equities + MCX Commodities):
    python test_breeze_mcx.py

    # 2. Run with live ICICI Direct Breeze credentials (if configured in .env):
    python test_breeze_mcx.py --live
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Windows Unicode console support
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai.advisor import AIAdvisor
from config.settings import BREEZE_API_KEY, BREEZE_SECRET_KEY, BREEZE_SESSION_TOKEN, STATIC_BREEZE_DIR
from core.universe import UniverseManager
from providers.breeze_provider import BreezeProvider
from providers.breeze_static_provider import BreezeStaticProvider
from screeners.breakdown import BreakdownScreener
from screeners.breakout import BreakoutScreener


def run_breeze_mcx_test(use_live: bool = False):
    print("=" * 80)
    print("🧪 ICICI DIRECT BREEZE API & MCX COMMODITIES SCREENING TEST")
    print("=" * 80)

    # 1. Determine Provider Mode
    if use_live:
        print("🔗 Mode: LIVE ICICI Direct Breeze API")
        if not (BREEZE_API_KEY and BREEZE_SECRET_KEY and BREEZE_SESSION_TOKEN):
            print("❌ Error: Missing BREEZE_API_KEY, BREEZE_SECRET_KEY, or BREEZE_SESSION_TOKEN in .env.")
            print("   Please configure them or run in offline test mode: python test_breeze_mcx.py")
            sys.exit(1)
        provider = BreezeProvider()
        if not provider.is_authenticated:
            print("❌ Error: Failed to authenticate with BreezeConnect. Check credentials or session token expiry.")
            sys.exit(1)
        print("✅ Successfully authenticated with Breeze API session!")
    else:
        print("📁 Mode: STATIC / OFFLINE BREEZE FIXTURES")
        print(f"   Directory: {STATIC_BREEZE_DIR}")
        provider = BreezeStaticProvider(static_dir=STATIC_BREEZE_DIR)
        print("✅ Static Breeze Provider initialized with sample fixtures.")

    # 2. Define Test Universe (Mix of MCX Commodities and NSE Equities)
    test_universe = [
        "GOLD",        # MCX Commodity: 52W High Breakout
        "TRENT",       # NSE Equity: Multi-week Resistance Breakout
        "CRUDEOIL",    # MCX Commodity: Support Breakdown / Stage-4
        "ASIANPAINT",  # NSE Equity: Support Breakdown / 52W Low testing
        "RELIANCE",    # NSE Equity: Neutral / Rangebound Control
        "SILVER"       # MCX Commodity: Upward Consolidation Control
    ]

    print(f"\n📋 Ingesting data for {len(test_universe)} assets across NSE & MCX:")
    for sym in test_universe:
        exchange = UniverseManager.get_exchange(sym)
        asset_type = "MCX Commodity" if exchange == "MCX" else "NSE Equity"
        print(f"   • {sym:<12} | Exchange: {exchange:<4} | Type: {asset_type}")

    # 3. Fetch Data
    t0 = time.time()
    data_dict = {}
    for sym in test_universe:
        df = provider.fetch_ohlcv(sym, period="1y", interval="1day")
        if df is not None and not df.empty:
            data_dict[sym] = df
            if use_live:
                provider.export_static_snapshot(sym, STATIC_BREEZE_DIR)

    fetch_time = time.time() - t0
    print(f"\n✅ Ingested {len(data_dict)} datasets in {fetch_time:.3f}s")

    # 4. Execute Screeners
    breakout_screener = BreakoutScreener()
    breakdown_screener = BreakdownScreener()

    breakout_candidates = []
    breakdown_candidates = []

    print("\n" + "=" * 80)
    print("🔍 RUNNING QUANTITATIVE SCREENERS")
    print("=" * 80)

    for sym, df in data_dict.items():
        exchange = UniverseManager.get_exchange(sym)
        # Test Breakout
        bo_result = breakout_screener.screen(sym, df)
        if bo_result:
            bo_result["exchange"] = exchange
            breakout_candidates.append(bo_result)

        # Test Breakdown
        bd_result = breakdown_screener.screen(sym, df)
        if bd_result:
            bd_result["exchange"] = exchange
            breakdown_candidates.append(bd_result)

    # 5. Report Results
    print(f"\n🚀 BREAKOUT CANDIDATES DETECTED: {len(breakout_candidates)}")
    print("-" * 80)
    for c in breakout_candidates:
        print(f"[{c['exchange']}] {c['symbol']} (Score: {c['score']}/100)")
        print(f"   LTP: ₹{c['close']} | Change: {c['change_pct']:+.2f}% | RVOL: {c['rvol']}x | RSI: {c['rsi']}")
        print(f"   Pivot: ₹{c['metrics']['breakout_pivot']} | 52W High: ₹{c['metrics']['52w_high']} (Distance: {c['metrics']['dist_from_52w_high_pct']}%)")
        print(f"   Triggers: {', '.join(c['reasons'])}\n")

    print(f"🔻 BREAKDOWN CANDIDATES DETECTED: {len(breakdown_candidates)}")
    print("-" * 80)
    for c in breakdown_candidates:
        print(f"[{c['exchange']}] {c['symbol']} (Score: {c['score']}/100 | Bias: {c['bias']})")
        print(f"   LTP: ₹{c['close']} | Change: {c['change_pct']:+.2f}% | RVOL: {c['rvol']}x | RSI: {c['rsi']}")
        print(f"   Support Pivot: ₹{c['metrics']['breakdown_pivot']} | 52W Low: ₹{c['metrics']['52w_low']} (Distance: {c['metrics']['dist_from_52w_low_pct']}%)")
        print(f"   Triggers: {', '.join(c['reasons'])}\n")

    # 6. AI Advisor Recommendations
    all_candidates = breakout_candidates + breakdown_candidates
    if all_candidates:
        print("=" * 80)
        print("🤖 AI TRADE ADVISOR (Actionable Trade Execution Plans)")
        print("=" * 80)
        advisor = AIAdvisor()
        status_msg = "Google Gemini Flash (Live AI)" if advisor.is_ai_ready else "Deterministic Mathematical Engine"
        print(f"Engine: {status_msg}...\n")

        plans = advisor.analyze_candidates(all_candidates)
        for p in plans:
            print(f"📌 [{p.get('category')}] {p.get('symbol')} ── Action: {p.get('trade_action')}")
            print(f"   Entry: ₹{p.get('entry_price')} | Stop Loss: ₹{p.get('stop_loss')}")
            print(f"   Target 1: ₹{p.get('target_1')} | Target 2: ₹{p.get('target_2')}")
            print(f"   Risk:Reward: {p.get('risk_reward_ratio')} | Conviction: {p.get('conviction')} | Timeframe: {p.get('timeframe')}")
            print(f"   Thesis: {p.get('thesis')}\n")

    # 7. Verification Summary
    print("=" * 80)
    print("🎯 VERIFICATION CHECKLIST")
    print("=" * 80)
    gold_passed = any(c["symbol"] == "GOLD" for c in breakout_candidates)
    trent_passed = any(c["symbol"] == "TRENT" for c in breakout_candidates)
    crude_passed = any(c["symbol"] == "CRUDEOIL" for c in breakdown_candidates)
    asian_passed = any(c["symbol"] == "ASIANPAINT" for c in breakdown_candidates)
    neutral_passed = not any(c["symbol"] in ["RELIANCE", "SILVER"] for c in all_candidates)

    print(f"  [{'PASS' if gold_passed else 'FAIL'}] MCX Commodity Breakout detected (GOLD)")
    print(f"  [{'PASS' if trent_passed else 'FAIL'}] NSE Equity Breakout detected (TRENT)")
    print(f"  [{'PASS' if crude_passed else 'FAIL'}] MCX Commodity Breakdown detected (CRUDEOIL)")
    print(f"  [{'PASS' if asian_passed else 'FAIL'}] NSE Equity Breakdown detected (ASIANPAINT)")
    print(f"  [{'PASS' if neutral_passed else 'FAIL'}] Control symbols filtered out (RELIANCE, SILVER - zero false positives)")

    all_tests_passed = gold_passed and trent_passed and crude_passed and asian_passed and neutral_passed
    if all_tests_passed:
        print("\n🎉 ALL TESTS PASSED! Breeze API & MCX Commodities pipeline is fully operational.")
    else:
        print("\n⚠️ Some test assertions failed. Please review candidate output.")

    return all_tests_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Breeze API & MCX Commodities Screening")
    parser.add_argument("--live", action="store_true", help="Connect to live ICICI Direct Breeze API (requires .env)")
    args = parser.parse_args()

    success = run_breeze_mcx_test(use_live=args.live)
    sys.exit(0 if success else 1)
