"""
Intraday Momentum Screener.
Screens for stocks showing unusual intraday activity, range expansion vs ATR, and trend momentum.
"""

from typing import Dict, Optional
import pandas as pd

from core.indicators import enrich_with_indicators
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class IntradayScreener(BaseScreener):
    """Detects high-momentum candidates for intraday day-trading."""

    def __init__(self):
        super().__init__(name="Intraday Momentum Screener", category="INTRADAY")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 20:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        volume = float(last['volume'])
        if close < 20.0 or volume < 100000:
            return None

        # 1. Day's Trading Range vs ATR
        day_range = float(last['high']) - float(last['low'])
        atr = float(last.get('atr_14', 1.0))
        range_to_atr = day_range / (atr + 1e-6)

        # 2. Relative Volume
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # 3. Trend Alignment
        ema9 = float(last.get('ema_9', 0))
        ema20 = float(last.get('ema_20', 0))
        clv = float(last.get('clv', 0.5))

        # Long Setup: Strong volume + range expansion + price > EMA9 >= EMA20 + CLV > 0.68
        is_long = (close >= ema9) and (ema9 >= ema20 * 0.99) and (clv >= 0.68) and (rvol >= 1.2 or rvol_9 >= 1.5) and (range_to_atr >= 0.5)
        
        # Short Setup: Strong volume + range expansion + price < EMA9 <= EMA20 + CLV < 0.32
        is_short = (close <= ema9) and (ema9 <= ema20 * 1.01) and (clv <= 0.32) and (rvol >= 1.2 or rvol_9 >= 1.5) and (range_to_atr >= 0.5)

        if is_long or is_short:
            direction = "BULLISH_INTRADAY" if is_long else "BEARISH_INTRADAY"
            score = 55.0

            # Retracement Check (50% Candle Body Defense)
            ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=is_long)
            is_retrace_healthy = ret_info["retracement_healthy"]
            retest_support_50pct = ret_info["retest_support_50pct"]

            # Sector Tailwind & Panic-Day Behavior
            tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=is_long)
            has_tailwind = tailwind["has_tailwind"]

            nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)
            panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)
            mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)

            if is_vol_confirmed_2x:
                score += 20.0
            elif rvol_9 >= 1.5 or rvol >= 1.6:
                score += 15.0
            elif rvol >= 1.3:
                score += 10.0

            if range_to_atr >= 0.8:
                score += 10.0
            elif range_to_atr >= 0.6:
                score += 5.0

            if (is_long and clv >= 0.80) or (is_short and clv <= 0.20):
                score += 10.0

            if is_retrace_healthy and retest_support_50pct > 0:
                score += 10.0

            if has_tailwind:
                score += tailwind["tailwind_score_bonus"]

            if panic_info["panic_resilient"] and is_long:
                score += panic_info["score_bonus"]

            score = min(score, 100.0)

            change_pct = ((close - float(prev['close'])) / float(prev['close'])) * 100.0
            reasons = [
                f"Direction: {direction}",
                f"Volume Confirmation: {rvol_9:.1f}x 9-day avg ({'> 2.0X' if is_vol_confirmed_2x else 'Elevated'})",
                f"Intraday range reached {range_to_atr * 100:.0f}% of daily ATR",
                f"CLV: {clv:.2f} ({'Buyers dominating close' if is_long else 'Sellers dominating'})"
            ]

            if is_retrace_healthy and retest_support_50pct > 0:
                reasons.append(ret_info["defense_desc"])

            if has_tailwind:
                reasons.append(tailwind["description"])

            if panic_info["panic_resilient"] and is_long:
                reasons.append(panic_info["description"])

            return {
                "symbol": symbol,
                "category": self.category,
                "bias": "LONG" if is_long else "SHORT",
                "score": round(score, 1),
                "close": round(close, 2),
                "change_pct": round(change_pct, 2),
                "rvol": round(rvol, 2),
                "rvol_9": round(rvol_9, 2),
                "vol_confirmed_2x": is_vol_confirmed_2x,
                "retest_support_50pct": retest_support_50pct,
                "retracement_healthy": is_retrace_healthy,
                "sector": tailwind["sector_name"],
                "has_sector_tailwind": has_tailwind,
                "panic_day_resilient": panic_info["panic_resilient"],
                "mansfield_rs": mrs_val,
                "rsi": round(float(last.get('rsi_14', 50.0)), 1),
                "reasons": reasons,
                "metrics": {
                    "day_range": round(day_range, 2),
                    "atr": round(atr, 2),
                    "range_to_atr": round(range_to_atr, 2),
                    "clv": round(clv, 3),
                    "day_high": round(float(last['high']), 2),
                    "day_low": round(float(last['low']), 2),
                    "rvol_9d": round(rvol_9, 2),
                    "vol_confirmation_2x": is_vol_confirmed_2x,
                    "retest_support_50pct": retest_support_50pct,
                    "retracement_defense": ret_info["defense_desc"],
                    "sector": tailwind["sector_name"],
                    "sector_tailwind": has_tailwind,
                    "panic_day_resilient": panic_info["panic_resilient"],
                    "mansfield_rs": mrs_val
                }
            }

        return None
