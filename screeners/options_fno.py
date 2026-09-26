"""
Options & F&O Screener (CE & PE Analysis).
Screens for high-probability Call (CE) and Put (PE) option buying setups
based on Price Action, Volatility, and Open Interest (OI) buildup dynamics.
"""

from typing import Dict, Optional
import pandas as pd

from core.indicators import enrich_with_indicators
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class OptionsScreener(BaseScreener):
    """Detects Call (CE) and Put (PE) trade opportunities."""

    def __init__(self, option_target: Optional[str] = None):
        super().__init__(name="Options CE/PE Buildup Screener", category="OPTIONS")
        self.option_target = option_target.upper() if option_target else None

    def screen(
        self,
        symbol: str,
        df: pd.DataFrame,
        oi_change_pct: Optional[float] = None,
        benchmark_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[dict]:
        if df is None or len(df) < 30:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        prev_close = float(prev['close'])
        price_change_pct = ((close - prev_close) / prev_close) * 100.0

        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        rsi = float(last.get('rsi_14', 50.0))
        atr = float(last.get('atr_14', close * 0.02))
        clv = float(last.get('clv', 0.5))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # Check for Call (CE) buying setup:
        # Bullish thrust, elevated volume, RSI momentum, strong close
        ce_setup = (price_change_pct >= 0.7) and (rvol >= 1.15 or rvol_9 >= 1.25) and (rsi >= 54.0) and (clv >= 0.65)
        
        # Check for Put (PE) buying setup:
        # Bearish break, elevated volume, RSI breakdown, weak close
        pe_setup = (price_change_pct <= -0.7) and (rvol >= 1.15 or rvol_9 >= 1.25) and (rsi <= 46.0) and (clv <= 0.35)

        # Apply specific filter if user requested only CE or only PE
        if self.option_target == "CE":
            pe_setup = False
        elif self.option_target == "PE":
            ce_setup = False

        # OI Buildup interpretation if OI change is provided
        buildup_type = "MOMENTUM_EXPANSION"
        if oi_change_pct is not None:
            if price_change_pct > 0 and oi_change_pct > 3.0:
                buildup_type = "LONG_BUILDUP"
                ce_setup = True if self.option_target != "PE" else False
            elif price_change_pct < 0 and oi_change_pct > 3.0:
                buildup_type = "SHORT_BUILDUP"
                pe_setup = True if self.option_target != "CE" else False
            elif price_change_pct > 0 and oi_change_pct < -2.0:
                buildup_type = "SHORT_COVERING"
            elif price_change_pct < 0 and oi_change_pct < -2.0:
                buildup_type = "LONG_UNWINDING"

        if ce_setup or pe_setup:
            option_type = "CE" if ce_setup else "PE"
            sentiment = "BULLISH" if ce_setup else "BEARISH"
            is_bull = ce_setup

            # Retracement Check (50% Candle Body Defense)
            ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=is_bull)
            is_retrace_healthy = ret_info["retracement_healthy"]
            retest_support_50pct = ret_info["retest_support_50pct"]

            # Sector Tailwind & Panic-Day Behavior
            tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=is_bull)
            has_tailwind = tailwind["has_tailwind"]

            nifty_df = benchmark_data.get("NIFTY") if benchmark_data else None
            if nifty_df is None and benchmark_data:
                nifty_df = benchmark_data.get("^NSEI")
            panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)
            mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)

            # Compute estimated ATM strike based on standard NSE strike intervals
            if close > 5000:
                step = 100.0
            elif close > 2000:
                step = 50.0
            elif close > 1000:
                step = 25.0
            elif close > 500:
                step = 10.0
            elif close > 200:
                step = 5.0
            else:
                step = 2.5

            atm_strike = round(close / step) * step
            itm_strike = atm_strike - step if option_type == "CE" else atm_strike + step
            otm_strike = atm_strike + step if option_type == "CE" else atm_strike - step

            score = 55.0
            reasons = []

            if abs(price_change_pct) >= 2.0:
                score += 15.0
            elif abs(price_change_pct) >= 1.0:
                score += 10.0

            # Volume Confirmation Scoring for Options
            if is_vol_confirmed_2x:
                score += 15.0
                reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x delta expansion)")
            elif rvol_9 >= 1.5:
                score += 12.0
                reasons.append(f"Strong volume expansion ({rvol_9:.1f}x 9-day avg)")
            elif rvol >= 1.25:
                score += 8.0

            if (ce_setup and clv >= 0.80) or (pe_setup and clv <= 0.20):
                score += 10.0

            if buildup_type in ["LONG_BUILDUP", "SHORT_BUILDUP"]:
                score += 10.0

            if is_retrace_healthy and retest_support_50pct > 0:
                score += 10.0
                reasons.append(ret_info["defense_desc"])

            if has_tailwind:
                score += tailwind["tailwind_score_bonus"]
                reasons.append(tailwind["description"])

            if is_bull and panic_info["panic_resilient"]:
                score += panic_info["score_bonus"]
                reasons.append(panic_info["description"])

            score = min(score, 100.0)

            reasons.append(f"Option Play: Buy {atm_strike:.0f} {option_type} (ATM)")
            reasons.append(f"Setup: {buildup_type} ({sentiment})")
            reasons.append(f"Price Change: {price_change_pct:+.2f}% with {rvol_9:.1f}x 9d volume")
            reasons.append(f"RSI: {rsi:.1f}, ATR: {atr:.2f}")

            return {
                "symbol": symbol,
                "category": self.category,
                "option_type": option_type,
                "recommended_strike": atm_strike,
                "strike_itm": itm_strike,
                "strike_otm": otm_strike,
                "score": round(score, 1),
                "close": round(close, 2),
                "change_pct": round(price_change_pct, 2),
                "rvol": round(rvol, 2),
                "rvol_9": round(rvol_9, 2),
                "vol_confirmed_2x": is_vol_confirmed_2x,
                "retest_support_50pct": retest_support_50pct,
                "retracement_healthy": is_retrace_healthy,
                "sector": tailwind["sector_name"],
                "has_sector_tailwind": has_tailwind,
                "panic_day_resilient": panic_info["panic_resilient"] if is_bull else False,
                "mansfield_rs": mrs_val,
                "rsi": round(rsi, 1),
                "reasons": reasons,
                "metrics": {
                    "atm_strike": atm_strike,
                    "itm_strike": itm_strike,
                    "otm_strike": otm_strike,
                    "buildup": buildup_type,
                    "atr": round(atr, 2),
                    "clv": round(clv, 3),
                    "rvol_9d": round(rvol_9, 2),
                    "vol_confirmation_2x": is_vol_confirmed_2x,
                    "retest_support_50pct": retest_support_50pct,
                    "retracement_defense": ret_info["defense_desc"],
                    "sector": tailwind["sector_name"],
                    "sector_tailwind": has_tailwind,
                    "panic_day_resilient": panic_info["panic_resilient"] if is_bull else False,
                    "mansfield_rs": mrs_val
                }
            }

        return None
