"""
BTST (Buy Today, Sell Tomorrow) Screener.
Screens for stocks with strong institutional close, volume breakout, and fresh 5-day highs.
"""

from typing import Dict, Optional
import pandas as pd

from core.indicators import enrich_with_indicators
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class BTSTScreener(BaseScreener):
    """Detects high-conviction BTST candidates for capturing overnight momentum."""

    def __init__(self):
        super().__init__(name="BTST Momentum Screener", category="BTST")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 30:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        volume = float(last['volume'])
        if close < 15.0 or volume < 50000:
            return None

        # 1. Close Location Value (CLV)
        clv = float(last.get('clv', 0.5))

        # 2. Relative Volume
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # 3. 5-Day High Breakout
        prev_5_high = float(df['high'].iloc[-6:-1].max()) if len(df) >= 6 else float(prev['high'])
        is_5d_breakout = close >= prev_5_high * 0.995

        # 4. Momentum & Trend
        rsi = float(last.get('rsi_14', 50.0))
        ema20 = float(last.get('ema_20', 0))
        is_uptrend = close >= ema20 * 0.99

        # 5. Retracement Check (50% Candle Body Defense Rule)
        ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=True)
        is_retrace_healthy = ret_info["retracement_healthy"]
        retest_support_50pct = ret_info["retest_support_50pct"]

        # 6. Sector Tailwind & Panic-Day Behavior
        tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=True)
        has_tailwind = tailwind["has_tailwind"]

        nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)
        panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)
        mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)

        # Basic eligibility: Positive session, closed strong in upper range
        is_positive_session = close >= float(prev['close'])
        cond_close_near_high = clv >= 0.70

        if is_positive_session and cond_close_near_high and is_uptrend:
            score = 50.0
            reasons = []

            if clv >= 0.85:
                score += 15.0
                reasons.append(f"Strong closing accumulation in top {(1.0-clv)*100:.0f}% of range (CLV {clv:.2f})")
            else:
                score += 5.0
                reasons.append(f"Firm finish near day high (CLV {clv:.2f})")

            # Volume Confirmation
            if is_vol_confirmed_2x:
                score += 20.0
                reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x)")
            elif rvol_9 >= 1.5:
                score += 15.0
                reasons.append(f"Volume surge of {rvol_9:.1f}x 9-day avg")
            elif rvol >= 1.2:
                score += 10.0
                reasons.append(f"Above-average volume ({rvol:.1f}x)")

            if is_5d_breakout:
                score += 10.0
                reasons.append("Testing or breaking fresh 5-day swing high")

            if 52.0 <= rsi <= 76.0:
                score += 10.0
                reasons.append(f"Healthy bullish RSI momentum ({rsi:.1f})")

            # Retracement Defense
            if is_retrace_healthy and retest_support_50pct > 0:
                reasons.append(ret_info["defense_desc"])
                score += 10.0

            # Sector Tailwind
            if has_tailwind:
                reasons.append(tailwind["description"])
                score += tailwind["tailwind_score_bonus"]

            # Panic-Day Resilience
            if panic_info["panic_resilient"]:
                reasons.append(panic_info["description"])
                score += panic_info["score_bonus"]

            if mrs_val > 0:
                reasons.append(f"Mansfield RS: {mrs_desc}")

            if score >= 60.0:
                score = min(score, 100.0)
                change_pct = ((close - float(prev['close'])) / float(prev['close'])) * 100.0

                return {
                    "symbol": symbol,
                    "category": self.category,
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
                    "rsi": round(rsi, 1),
                    "reasons": reasons,
                    "metrics": {
                        "clv": round(clv, 3),
                        "prev_5d_high": round(prev_5_high, 2),
                        "day_high": round(float(last['high']), 2),
                        "day_low": round(float(last['low']), 2),
                        "ema_20": round(ema20, 2),
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
