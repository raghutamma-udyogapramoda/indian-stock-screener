"""
Swing Trading & Value Pullback Screener.
Screens for stocks in established Stage-2 uptrends pulling back to key support zones
(20 EMA, 50 EMA, or lower Bollinger Band) and forming bullish reversal price action.
Features high risk-to-reward setups (1:2.5+) with tight stop-losses for 1-4 week swing trades.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class SwingScreener(BaseScreener):
    """Detects high-probability swing trading and pullback setups."""

    def __init__(self):
        super().__init__(name="Swing Pullback & Reversal Screener", category="SWING")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 50:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        open_price = float(last['open'])
        high = float(last['high'])
        low = float(last['low'])
        volume = float(last['volume'])
        if close < 15.0 or volume < 25000:
            return None

        ema20 = float(last.get('ema_20', 0))
        ema50 = float(last.get('ema_50', 0))
        sma200 = float(last.get('sma_200', 0)) if 'sma_200' in df.columns and not np.isnan(last['sma_200']) else ema50
        atr = float(last.get('atr_14', close * 0.02))
        rsi = float(last.get('rsi_14', 50.0))
        clv = float(last.get('clv', 0.5))
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        is_vol_confirmed_2x = rvol_9 >= 2.0

        # 1. Trend Structure: Uptrend bias (EMA20 >= EMA50 or Price >= SMA200)
        is_bullish_structure = (ema20 >= ema50 * 0.98) or (close >= sma200)

        # 2. Pullback Proximity to Key Support:
        # Distance from EMA 20 or EMA 50 within 2.5%
        dist_ema20_pct = abs((close - ema20) / (ema20 + 1e-6)) * 100.0
        dist_ema50_pct = abs((close - ema50) / (ema50 + 1e-6)) * 100.0
        near_key_ema = (dist_ema20_pct <= 2.8) or (dist_ema50_pct <= 2.8)

        # Lower Bollinger Band touch / bounce in last 3 days
        bb_lower = float(last.get('bb_lower', 0))
        near_bb_lower = low <= (bb_lower * 1.02)

        is_pullback = near_key_ema or near_bb_lower

        # 3. Bullish Reversal & Momentum Confirmation
        # Candle body and shadows
        body = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        is_hammer = (lower_shadow >= 1.5 * (body + 1e-6)) and (clv >= 0.50)
        is_bullish_close = (close >= open_price) and (clv >= 0.55)

        prev_open = float(prev['open'])
        prev_close = float(prev['close'])
        is_engulfing = (close > prev_open) and (open_price <= prev_close) and (close > prev_close) and (close > open_price)

        macd_line = float(last.get('macd_line', 0))
        macd_signal = float(last.get('macd_signal', 0))
        macd_level2 = (macd_line > macd_signal) and (macd_line > 0)

        aroon_up = float(last.get('aroon_up', 50))
        aroon_osc = float(last.get('aroon_osc', 0))
        aroon_bullish = (aroon_osc >= 20.0) or (aroon_up >= 70.0)

        # 4. Healthy RSI Momentum Zone (40 to 62: healthy pullback, not deeply broken)
        healthy_rsi = 38.0 <= rsi <= 64.0

        if is_bullish_structure and is_pullback and (is_bullish_close or is_hammer or is_engulfing) and healthy_rsi:
            score = 60.0
            reasons = []

            if near_key_ema:
                support_name = "EMA 20" if dist_ema20_pct < dist_ema50_pct else "EMA 50"
                reasons.append(f"Pullback to high-value dynamic support ({support_name})")
                score += 10.0

            if near_bb_lower:
                reasons.append("Mean reversion bounce off lower Bollinger Band")
                score += 10.0

            if is_hammer:
                reasons.append("Hammer / Long lower shadow showing buyer absorption")
                score += 15.0
            elif is_engulfing:
                reasons.append("Bullish Engulfing pattern (Resistance engulfed today)")
                score += 15.0
            elif clv >= 0.70:
                reasons.append(f"Strong closing momentum off the lows (CLV {clv:.2f})")
                score += 10.0

            # Volume Confirmation at Support
            if is_vol_confirmed_2x:
                score += 15.0
                reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x buyer absorption at support)")
            elif rvol_9 >= 1.5:
                score += 10.0
                reasons.append(f"Solid demand volume ({rvol_9:.1f}x 9-day avg)")
            elif rvol >= 1.2:
                score += 5.0
                reasons.append(f"Above-average support volume ({rvol:.1f}x)")

            if macd_level2:
                reasons.append("MACD Level-II bullish continuation (MACD > Signal above zero line)")
                score += 5.0

            if aroon_bullish:
                reasons.append(f"Aroon Oscillator strong buy signal (Osc: +{aroon_osc:.0f})")
                score += 5.0

            if 42.0 <= rsi <= 55.0:
                reasons.append(f"Optimal swing entry zone (RSI reset to {rsi:.1f})")
                score += 10.0

            # Retracement Check (50% Candle Body Defense Rule)
            ret_info = MarketIntelAnalyzer.evaluate_retracement(df, is_bullish=True)
            is_retrace_healthy = ret_info["retracement_healthy"]
            retest_support_50pct = ret_info["retest_support_50pct"]
            if is_retrace_healthy and retest_support_50pct > 0:
                reasons.append(ret_info["defense_desc"])
                score += 10.0

            # Sector Tailwind & Panic-Day Behavior
            tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=True)
            has_tailwind = tailwind["has_tailwind"]
            if has_tailwind:
                reasons.append(tailwind["description"])
                score += tailwind["tailwind_score_bonus"]

            nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)
            panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)
            mrs_val, mrs_desc = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df)
            if panic_info["panic_resilient"]:
                reasons.append(panic_info["description"])
                score += panic_info["score_bonus"]

            if mrs_val > 0:
                reasons.append(f"Mansfield RS: {mrs_desc}")

            score = min(score, 100.0)

            # Swing Levels
            # Stop Loss below lowest of last 3 bars or 50% candle body support
            recent_low = float(df['low'].iloc[-4:].min())
            effective_support = min(recent_low, retest_support_50pct) if retest_support_50pct > 0 else recent_low
            stop_loss = round(min(effective_support, close - (1.5 * atr)), 2)

            # Target at 20-day swing high
            recent_20_high = float(df['high'].iloc[-21:-1].max()) if len(df) >= 21 else float(high * 1.05)
            target = round(max(recent_20_high, close + (2.5 * (close - stop_loss))), 2)

            risk = close - stop_loss
            reward = target - close
            rr = reward / (risk + 1e-6)

            change_pct = ((close - float(prev['close'])) / float(prev['close'])) * 100.0

            return {
                "symbol": symbol,
                "category": self.category,
                "bias": "LONG",
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
                    "support_level": round(stop_loss, 2),
                    "target_level": round(target, 2),
                    "risk_reward_est": f"1:{rr:.1f}",
                    "ema_20": round(ema20, 2),
                    "ema_50": round(ema50, 2),
                    "clv": round(clv, 3),
                    "atr": round(atr, 2),
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
