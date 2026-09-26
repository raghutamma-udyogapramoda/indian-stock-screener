"""
Weekend & Multi-Week Breakout Screener.
Screens for stocks in Stage-2 uptrend coiling in tight volatility squeezes near 52-week or multi-month resistance.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.universe import UniverseManager
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class BreakoutScreener(BaseScreener):
    """Detects high-probability positional breakout candidates."""

    def __init__(self):
        super().__init__(name="Multi-Week / 52W Breakout Screener", category="BREAKOUT")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 60:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        volume = float(last['volume'])
        
        is_comm = UniverseManager.is_commodity(symbol)
        min_vol = 50.0 if is_comm else 20000.0
        if close < 15.0 or volume < min_vol:
            return None

        # 1. 52-Week & Multi-Week Highs
        lookback_len = min(252, len(df))
        highest_high = float(df['high'].iloc[-lookback_len:].max())
        dist_from_high_pct = ((highest_high - close) / highest_high) * 100.0

        # Multi-week (20-day) swing resistance & Box consolidation
        prev_20_high = float(df['high'].iloc[-21:-1].max()) if len(df) >= 21 else float(prev['high'])
        prev_20_low = float(df['low'].iloc[-21:-1].min()) if len(df) >= 21 else float(prev['low'])
        box_height_pct = ((prev_20_high - prev_20_low) / (prev_20_low + 1e-6)) * 100.0
        dist_from_20d_high_pct = ((prev_20_high - close) / (prev_20_high + 1e-6)) * 100.0
        is_20d_breakout = close > prev_20_high
        is_box_breakout_soon = (dist_from_20d_high_pct <= 5.0) and (box_height_pct <= 12.0)

        # 2. Stage-2 Trend Filter
        ema20 = float(last.get('ema_20', 0))
        ema50 = float(last.get('ema_50', 0))
        sma200 = float(last.get('sma_200', 0)) if 'sma_200' in df.columns and not np.isnan(last['sma_200']) else ema50

        is_uptrend = (close >= ema20) and (ema20 >= ema50 * 0.98)

        # 3. Bollinger Volatility Squeeze
        bbw = df['bb_bandwidth'].iloc[-40:]
        current_bbw = float(last['bb_bandwidth'])
        min_bbw_40 = float(bbw.min())
        is_squeezing = current_bbw <= (min_bbw_40 * 1.35)  # within 35% of lowest volatility

        # 4. Momentum & Candlestick Enhancements (MACD Level-II, Aroon, Engulfing)
        macd_line = float(last.get('macd_line', 0))
        macd_signal = float(last.get('macd_signal', 0))
        macd_level2 = (macd_line > macd_signal) and (macd_line > 0)

        aroon_up = float(last.get('aroon_up', 50))
        aroon_osc = float(last.get('aroon_osc', 0))
        aroon_bullish = (aroon_osc >= 30.0) or (aroon_up >= 70.0)

        open_p = float(last['open'])
        prev_o = float(prev['open'])
        prev_c = float(prev['close'])
        is_engulfing = (close > prev_o) and (open_p <= prev_c) and (close > prev_c) and (close > open_p)

        # 5. Fresh Breakout or Imminent Testing
        is_near_52w = dist_from_high_pct <= 4.0  # within 4% of 52W high
        is_fresh_52w = close >= highest_high * 0.99
        is_breakout_candidate = (
            is_fresh_52w
            or is_near_52w
            or (is_20d_breakout and close > prev_c)
            or (is_box_breakout_soon and (is_squeezing or aroon_bullish or macd_level2))
        )

        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        clv = float(last.get('clv', 0.5))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # 6. Retracement Check: 50% Candle Body Defense Rule
        ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=True)
        is_retrace_healthy = ret_info["retracement_healthy"]
        retest_support_50pct = ret_info["retest_support_50pct"]

        # 7. Sector Tailwind & Panic-Day Behavior
        tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=True)
        has_tailwind = tailwind["has_tailwind"]

        nifty_df = benchmark_data.get("NIFTY") if benchmark_data else None
        if nifty_df is None and benchmark_data:
            nifty_df = benchmark_data.get("^NSEI")
        panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)
        mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)

        reasons = []
        if is_fresh_52w:
            reasons.append("Breaking out to 52-week / All-Time high")
        elif is_near_52w:
            reasons.append(f"Testing key 52W resistance ({dist_from_high_pct:.1f}% from 52W High)")
        elif is_20d_breakout:
            reasons.append(f"Multi-week resistance breakout (cleared 20-day high ₹{prev_20_high:.1f})")
        elif is_box_breakout_soon:
            reasons.append(f"Box Breakout on Watch (coiling within {dist_from_20d_high_pct:.1f}% of ₹{prev_20_high:.1f} resistance)")

        if is_squeezing:
            reasons.append("Bollinger Band squeeze (volatility contraction expansion)")

        if is_uptrend:
            reasons.append("Stage-2 Trend alignment (Price > EMA20 >= EMA50)")

        if macd_level2:
            reasons.append("MACD Level-II bullish momentum (MACD > Signal above zero line)")

        if aroon_bullish:
            reasons.append(f"Aroon Oscillator strong buy signal (Osc: +{aroon_osc:.0f})")

        if is_engulfing:
            reasons.append("Bullish Engulfing pattern (Resistance engulfed today)")

        # Institutional Volume Confirmation
        if is_vol_confirmed_2x:
            reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x)")
        elif rvol_9 >= 1.5:
            reasons.append(f"Strong volume expansion ({rvol_9:.1f}x 9-day avg)")
        elif rvol >= 1.25:
            reasons.append(f"Volume surge ({rvol:.1f}x 20-day avg)")

        # Retracement Check Reason
        if is_retrace_healthy and retest_support_50pct > 0:
            reasons.append(ret_info["defense_desc"])

        # Sector Tailwind Reason
        if has_tailwind:
            reasons.append(tailwind["description"])

        # Panic-Day Resilience Reason
        if panic_info["panic_resilient"]:
            reasons.append(panic_info["description"])

        if mrs_val > 0:
            reasons.append(f"Mansfield RS: {mrs_desc}")

        # Screener qualification threshold
        if is_breakout_candidate and is_uptrend and (clv >= 0.50):
            # Score from 0 to 100
            score = 55.0
            if is_fresh_52w:
                score += 15.0
            elif is_20d_breakout:
                score += 10.0
            elif is_box_breakout_soon:
                score += 8.0

            if is_squeezing:
                score += 8.0
            if macd_level2:
                score += 5.0
            if aroon_bullish:
                score += 5.0
            if is_engulfing:
                score += 5.0

            # Volume Confirmation Scoring
            if is_vol_confirmed_2x:
                score += 15.0
            elif rvol_9 >= 1.5 or rvol >= 1.3:
                score += 10.0

            # Retracement Defense Scoring (+10 pts)
            if is_retrace_healthy:
                score += 10.0

            # Sector Tailwind Scoring (+10 pts)
            if has_tailwind:
                score += tailwind["tailwind_score_bonus"]

            # Panic-Day Resilience Scoring (+15 pts)
            if panic_info["panic_resilient"]:
                score += panic_info["score_bonus"]

            if clv >= 0.75:
                score += 8.0

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
                "rsi": round(float(last.get('rsi_14', 50.0)), 1),
                "reasons": reasons,
                "metrics": {
                    "breakout_pivot": round(prev_20_high, 2),
                    "dist_from_52w_high_pct": round(dist_from_high_pct, 2),
                    "52w_high": round(highest_high, 2),
                    "bb_bandwidth": round(current_bbw, 4),
                    "ema_20": round(ema20, 2),
                    "ema_50": round(ema50, 2),
                    "sma_200": round(sma200, 2),
                    "macd_hist": round(float(last.get('macd_hist', 0)), 2),
                    "aroon_osc": round(aroon_osc, 1),
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
