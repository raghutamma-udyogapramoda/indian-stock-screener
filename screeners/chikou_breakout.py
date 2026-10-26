"""
Ichimoku Chikou Span Breakout Screener.
Implements the signature 'ONE setup for life' popularized by technical trader Manjunath S. (@ChikouTrader).
Screens for high-probability momentum breakouts where price is above the Kumo Cloud and the Chikou Span
(Lagging Span) clears into unobstructed 'Open Space' above historical candles from 26 periods ago.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.universe import UniverseManager
from core.relative_strength import MarketIntelAnalyzer
from screeners.base_screener import BaseScreener


class ChikouBreakoutScreener(BaseScreener):
    """Detects high-conviction Ichimoku Kumo and Chikou Span Open Space breakouts."""

    def __init__(self):
        super().__init__(name="Ichimoku Chikou Breakout Screener", category="CHIKOU_BREAKOUT")

    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        if df is None or len(df) < 52:
            return None

        df = enrich_with_indicators(df.copy())
        if "tenkan_sen" not in df.columns or "kumo_top" not in df.columns:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        volume = float(last['volume'])

        is_comm = UniverseManager.is_commodity(symbol) or UniverseManager.is_global_commodity(symbol)
        min_vol = 50.0 if is_comm else 20000.0
        min_price = 0.5 if is_comm else 15.0
        if close < min_price or volume < min_vol:
            return None

        # 1. Ichimoku Core Components
        tenkan = float(last.get('tenkan_sen', 0.0))
        kijun = float(last.get('kijun_sen', 0.0))
        kumo_top = float(last.get('kumo_top', 0.0))
        kumo_bottom = float(last.get('kumo_bottom', 0.0))
        span_a = float(last.get('senkou_span_a', 0.0))
        span_b = float(last.get('senkou_span_b', 0.0))
        future_a = float(last.get('future_span_a', 0.0))
        future_b = float(last.get('future_span_b', 0.0))

        past_high_26 = float(last.get('past_high_26', 0.0))
        past_close_26 = float(last.get('past_close_26', 0.0))
        chikou_open_space = bool(last.get('chikou_open_space', False))
        chikou_clearance_pct = float(last.get('chikou_clearance_pct', 0.0))

        if kumo_top <= 0 or past_high_26 <= 0 or kijun <= 0:
            return None

        # 2. Setup Conditions:
        # Condition A: Price cleanly above Kumo Cloud (or breaking out on current bar)
        price_above_kumo = close >= (kumo_top * 0.995)
        dist_above_kumo_pct = ((close - kumo_top) / kumo_top) * 100.0

        # Condition B: Tenkan-Sen >= Kijun-Sen (Bullish TK alignment or crossover)
        tk_bullish = tenkan >= (kijun * 0.995)
        price_above_tenkan = close >= (tenkan * 0.99)

        # Condition C: Chikou Span in Open Space (Current close clearing bar 26's high with open headroom)
        chikou_cleared = (close > past_high_26) or chikou_open_space

        # Condition D: Forward Cloud Bullish confirmation
        future_kumo_bullish = future_a >= future_b

        # Filter out setups that fail any core Ichimoku pillar
        if not (price_above_kumo and tk_bullish and chikou_cleared):
            return None

        # 3. Volume and Candlestick Quality
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        clv = float(last.get('clv', 0.5))
        if clv < 0.40 and rvol < 0.9:
            return None

        # 4. Sector Tailwind & Panic-Day Resilience
        tailwind = MarketIntelAnalyzer.evaluate_sector_tailwind(symbol, benchmark_data, is_bullish=True)
        has_tailwind = tailwind["has_tailwind"]
        nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)
        panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df)

        # 5. Risk Management: Stop Loss and Targets
        # Stop loss anchored at Kijun-sen (standard Ichimoku exit) or recent 5-day low
        recent_5_low = float(df['low'].iloc[-5:].min())
        stop_loss_candidate = max(kijun, recent_5_low)
        # Cap SL between 2.0% and 6.5% for robust risk-reward
        min_sl = close * 0.935
        max_sl = close * 0.98
        stop_loss = max(min_sl, min(stop_loss_candidate, max_sl))
        risk_pct = ((close - stop_loss) / close) * 100.0

        target_1 = round(close + (close - stop_loss) * 1.5, 2)
        target_2 = round(close + (close - stop_loss) * 2.5, 2)
        risk_reward = round((target_1 - close) / max(0.01, close - stop_loss), 2)

        # 6. Scoring System (50 to 100)
        score = 55.0
        if chikou_open_space:
            score += 15.0
        elif chikou_cleared:
            score += 10.0

        if chikou_clearance_pct >= 2.0:
            score += 5.0

        if dist_above_kumo_pct >= 2.0:
            score += 10.0

        if future_kumo_bullish:
            score += 5.0

        if tenkan > kijun:
            score += 5.0

        if rvol >= 1.3 or rvol_9 >= 1.5:
            score += 8.0

        if clv >= 0.70:
            score += 5.0

        if has_tailwind:
            score += tailwind["tailwind_score_bonus"]

        if panic_info["panic_resilient"]:
            score += panic_info["score_bonus"]

        score = min(round(score, 1), 100.0)

        # 7. Reasons & Descriptive Thesis
        reasons = []
        if chikou_open_space:
            reasons.append(f"Chikou in Open Space (+{chikou_clearance_pct:.1f}% above past resistance)")
        else:
            reasons.append(f"Chikou cleared candle 26 bars ago (+{chikou_clearance_pct:.1f}%)")

        reasons.append(f"Price above Kumo Cloud (+{dist_above_kumo_pct:.1f}%)")

        if tenkan > kijun:
            reasons.append("Bullish Tenkan-Kijun alignment")

        if future_kumo_bullish:
            reasons.append("Forward Kumo Cloud expanding green")

        if rvol >= 1.3:
            reasons.append(f"Expanding institutional volume ({rvol:.2f}x 20-day SMA)")

        if has_tailwind:
            reasons.append(f"Sector tailwind in {tailwind['sector_name']}")

        change_pct = ((close - float(prev['close'])) / float(prev['close'])) * 100.0

        return {
            "symbol": symbol,
            "category": self.category,
            "screener_name": self.name,
            "score": score,
            "close": round(close, 2),
            "change_pct": round(change_pct, 2),
            "rvol": round(rvol, 2),
            "rvol_9": round(rvol_9, 2),
            "trigger_price": round(close, 2),
            "stop_loss": round(stop_loss, 2),
            "target_1": target_1,
            "target_2": target_2,
            "risk_reward": risk_reward,
            "risk_pct": round(risk_pct, 2),
            "chikou_open_space": chikou_open_space,
            "chikou_clearance_pct": round(chikou_clearance_pct, 2),
            "dist_above_kumo_pct": round(dist_above_kumo_pct, 2),
            "future_kumo_bullish": future_kumo_bullish,
            "reasons": reasons,
            "metrics": {
                "tenkan_sen": round(tenkan, 2),
                "kijun_sen": round(kijun, 2),
                "kumo_top": round(kumo_top, 2),
                "kumo_bottom": round(kumo_bottom, 2),
                "past_high_26": round(past_high_26, 2),
                "chikou_clearance_pct": round(chikou_clearance_pct, 2),
                "dist_above_kumo_pct": round(dist_above_kumo_pct, 2),
                "future_kumo_bullish": future_kumo_bullish,
                "rvol_20": round(rvol, 2),
                "sector": tailwind["sector_name"],
                "has_sector_tailwind": has_tailwind,
                "panic_day_resilient": panic_info["panic_resilient"],
            }
        }
