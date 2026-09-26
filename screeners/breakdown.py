"""
Breakdown & Distribution Screener.
Screens for stocks in Stage-4 downtrends breaking down below key multi-week support,
20-day swing lows, or 52-week lows on heavy institutional selling volume.
Ideal for Short selling, Intraday shorts, Put (PE) buying, and Bearish Swing setups.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.universe import UniverseManager
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class BreakdownScreener(BaseScreener):
    """Detects high-conviction breakdown and short-selling candidates."""

    def __init__(self):
        super().__init__(name="Multi-Week / Support Breakdown Screener", category="BREAKDOWN")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 40:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        volume = float(last['volume'])
        
        is_comm = UniverseManager.is_commodity(symbol)
        min_vol = 50.0 if is_comm else 25000.0
        if close < 15.0 or volume < min_vol:
            return None

        # 1. Multi-week & 52-Week Lows
        lookback_len = min(252, len(df))
        lowest_low_52w = float(df['low'].iloc[-lookback_len:].min())
        dist_from_low_pct = ((close - lowest_low_52w) / (lowest_low_52w + 1e-6)) * 100.0

        # 20-day swing low (prior to current candle)
        prev_20_low = float(df['low'].iloc[-21:-1].min()) if len(df) >= 21 else float(prev['low'])
        is_20d_breakdown = close < prev_20_low
        is_52w_breakdown = close <= lowest_low_52w * 1.01 or dist_from_low_pct <= 3.5

        # 2. Stage-4 Downtrend Filter
        ema20 = float(last.get('ema_20', 0))
        ema50 = float(last.get('ema_50', 0))
        sma200 = float(last.get('sma_200', 0)) if 'sma_200' in df.columns and not np.isnan(last['sma_200']) else ema50

        is_downtrend = (close <= ema20) and (ema20 <= ema50)

        # 3. Selling Pressure & Price Action
        clv = float(last.get('clv', 0.5))
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        rsi = float(last.get('rsi_14', 50.0))
        atr = float(last.get('atr_14', close * 0.02))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # 4. Retracement Check (50% Breakdown Candle Body Ceiling Rule)
        ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=False)
        is_retrace_healthy = ret_info["retracement_healthy"]
        retest_resistance_50pct = ret_info["retest_support_50pct"]

        # 5. Sector Tailwind (Downward Pressure) & Relative Weakness
        tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=False)
        has_tailwind = tailwind["has_tailwind"]

        nifty_df = benchmark_data.get("NIFTY") if benchmark_data else None
        if nifty_df is None and benchmark_data:
            nifty_df = benchmark_data.get("^NSEI")
        mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)

        # Check conditions
        cond_support_breach = is_20d_breakdown or is_52w_breakdown
        cond_heavy_selling = (clv <= 0.35) or (close < float(prev['close']))
        cond_bearish_momentum = rsi <= 48.0

        if cond_support_breach and is_downtrend and cond_heavy_selling and cond_bearish_momentum:
            score = 55.0
            reasons = []

            if is_52w_breakdown:
                score += 15.0
                reasons.append(f"Testing or breaking 52-week low ({dist_from_low_pct:.1f}% from 52W Low)")
            elif is_20d_breakdown:
                score += 10.0
                reasons.append("Violated key 20-day swing support")

            if clv <= 0.20:
                score += 10.0
                reasons.append(f"Sellers dominated close (CLV {clv:.2f})")

            # Volume Confirmation Scoring
            if is_vol_confirmed_2x:
                score += 15.0
                reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x dumping)")
            elif rvol_9 >= 1.5:
                score += 12.0
                reasons.append(f"Heavy selling volume ({rvol_9:.1f}x 9-day avg)")
            elif rvol >= 1.3:
                score += 8.0
                reasons.append(f"Elevated selling volume ({rvol:.1f}x 20-day avg)")

            if rsi <= 35.0:
                score += 5.0
                reasons.append(f"Bearish momentum acceleration (RSI {rsi:.1f})")

            # Retracement Ceiling Defense
            if is_retrace_healthy and retest_resistance_50pct > 0:
                score += 10.0
                reasons.append(ret_info["defense_desc"])

            # Sector Downward Pressure
            if has_tailwind:
                score += tailwind["tailwind_score_bonus"]
                reasons.append(tailwind["description"])

            if mrs_val < 0:
                reasons.append(f"Severe Relative Weakness vs NIFTY (Mansfield RS: {mrs_val:.1f}%)")

            reasons.append("Stage-4 downtrend alignment (Price < EMA20 < EMA50)")
            score = min(score, 100.0)

            change_pct = ((close - float(prev['close'])) / float(prev['close'])) * 100.0

            return {
                "symbol": symbol,
                "category": self.category,
                "bias": "SHORT",
                "score": round(score, 1),
                "close": round(close, 2),
                "change_pct": round(change_pct, 2),
                "rvol": round(rvol, 2),
                "rvol_9": round(rvol_9, 2),
                "vol_confirmed_2x": is_vol_confirmed_2x,
                "retest_support_50pct": retest_resistance_50pct,
                "retracement_healthy": is_retrace_healthy,
                "sector": tailwind["sector_name"],
                "has_sector_tailwind": has_tailwind,
                "panic_day_resilient": False,
                "mansfield_rs": mrs_val,
                "rsi": round(rsi, 1),
                "reasons": reasons,
                "metrics": {
                    "breakdown_pivot": round(prev_20_low, 2),
                    "52w_low": round(lowest_low_52w, 2),
                    "dist_from_52w_low_pct": round(dist_from_low_pct, 2),
                    "clv": round(clv, 3),
                    "ema_20": round(ema20, 2),
                    "ema_50": round(ema50, 2),
                    "atr": round(atr, 2),
                    "rvol_9d": round(rvol_9, 2),
                    "vol_confirmation_2x": is_vol_confirmed_2x,
                    "retest_resistance_50pct": retest_resistance_50pct,
                    "retracement_defense": ret_info["defense_desc"],
                    "sector": tailwind["sector_name"],
                    "sector_tailwind": has_tailwind,
                    "mansfield_rs": mrs_val
                }
            }

        return None
