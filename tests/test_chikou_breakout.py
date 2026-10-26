"""
Unit Tests for Ichimoku Chikou Breakout Screener and Backtesting Strategy.
Validates Ichimoku Kinko Hyo calculations, Chikou Open Space clearance rules,
screener filtering and ranking, and backtesting strategy integration.
"""

import unittest
import numpy as np
import pandas as pd

from core.indicators import compute_ichimoku, enrich_with_indicators
from screeners.chikou_breakout import ChikouBreakoutScreener
from core.backtesting.strategy import BuiltinStrategies, ChikouBreakoutStrategy, StrategyConfig
from core.backtesting.evaluator import enrich_plans_with_backtest_evaluations


class TestChikouBreakout(unittest.TestCase):
    """Test suite for Ichimoku Chikou Breakout system."""

    def _generate_synthetic_df(self, n_bars: int = 120, trend: str = "bullish") -> pd.DataFrame:
        dates = pd.date_range("2026-01-01", periods=n_bars, freq="D")
        np.random.seed(42)

        if trend == "bullish":
            # Consolidation for first 60 bars between 100-110, then explosive breakout to 140+
            if n_bars <= 60:
                base_prices = np.linspace(100, 140, n_bars)
            else:
                base_prices = np.concatenate([
                    np.linspace(100, 105, 60),
                    np.linspace(106, 145, n_bars - 60)
                ])
        elif trend == "bearish":
            # Consolidation then breakdown
            if n_bars <= 60:
                base_prices = np.linspace(130, 90, n_bars)
            else:
                base_prices = np.concatenate([
                    np.linspace(130, 125, 60),
                    np.linspace(124, 90, n_bars - 60)
                ])
        else:
            # Sideways chop
            base_prices = np.full(n_bars, 100.0) + np.sin(np.linspace(0, 10, n_bars)) * 3.0

        noise = np.random.normal(0, 0.8, n_bars)
        close = base_prices + noise
        high = close + np.random.uniform(0.5, 2.0, n_bars)
        low = close - np.random.uniform(0.5, 2.0, n_bars)
        open_p = (high + low) / 2.0
        volume = np.random.uniform(50000, 150000, n_bars)
        # Expand volume on final breakout bars
        volume[-5:] *= 2.5

        return pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        }, index=dates)

    def test_compute_ichimoku_elements(self):
        """Tests that compute_ichimoku calculates all core elements and open space status."""
        df = self._generate_synthetic_df(n_bars=80, trend="bullish")
        ich_df = compute_ichimoku(df["high"], df["low"], df["close"])

        expected_cols = [
            "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b",
            "kumo_top", "kumo_bottom", "future_span_a", "future_span_b",
            "past_high_26", "past_close_26", "past_low_26",
            "chikou_open_space", "chikou_clearance_pct"
        ]
        for col in expected_cols:
            self.assertIn(col, ich_df.columns, f"Missing expected column {col}")

        # Final candle in a strong bullish rally should have Chikou in open space
        last = ich_df.iloc[-1]
        self.assertTrue(last["tenkan_sen"] > 0)
        self.assertTrue(last["kijun_sen"] > 0)
        self.assertTrue(last["kumo_top"] >= last["kumo_bottom"])

    def test_enrich_with_indicators_includes_ichimoku(self):
        """Tests that enrich_with_indicators injects Ichimoku metrics onto OHLCV DataFrames."""
        df = self._generate_synthetic_df(n_bars=70, trend="bullish")
        enriched = enrich_with_indicators(df)

        self.assertIn("tenkan_sen", enriched.columns)
        self.assertIn("kumo_top", enriched.columns)
        self.assertIn("chikou_open_space", enriched.columns)

    def test_chikou_breakout_screener_qualifies_bullish_stock(self):
        """Tests that a stock with clear Chikou Open Space and Price > Kumo qualifies."""
        df = self._generate_synthetic_df(n_bars=100, trend="bullish")
        screener = ChikouBreakoutScreener()
        res = screener.screen("TEST_STOCK", df)

        self.assertIsNotNone(res, "Bullish Chikou setup should qualify")
        self.assertEqual(res["category"], "CHIKOU_BREAKOUT")
        self.assertGreaterEqual(res["score"], 50.0)
        self.assertIn("trigger_price", res)
        self.assertIn("stop_loss", res)
        self.assertIn("target_1", res)
        self.assertIn("target_2", res)
        self.assertTrue(res["stop_loss"] < res["close"], "Stop loss must be below close")
        self.assertTrue(res["target_1"] > res["close"], "Target must be above close")

    def test_chikou_breakout_screener_rejects_bearish_stock(self):
        """Tests that a stock trading below Kumo Cloud is rejected."""
        df = self._generate_synthetic_df(n_bars=100, trend="bearish")
        screener = ChikouBreakoutScreener()
        res = screener.screen("BEAR_STOCK", df)

        self.assertIsNone(res, "Stock in downtrend below Kumo should not qualify")

    def test_chikou_breakout_screener_rejects_short_history(self):
        """Tests that DataFrames with fewer than 52 bars are rejected safely."""
        df = self._generate_synthetic_df(n_bars=40, trend="bullish")
        screener = ChikouBreakoutScreener()
        res = screener.screen("SHORT_HISTORY", df)

        self.assertIsNone(res, "Insufficient history for 52-period Span B must return None")

    def test_chikou_breakout_backtest_strategy_signal(self):
        """Tests that ChikouBreakoutStrategy generates signals when setup criteria are met."""
        df = self._generate_synthetic_df(n_bars=100, trend="bullish")
        enriched = enrich_with_indicators(df)
        strat = ChikouBreakoutStrategy(StrategyConfig(segment="FUTURES"))

        class MockPortfolio:
            cash = 500000.0

        signal = strat.generate_signals(enriched, MockPortfolio(), as_of_idx=len(enriched) - 1)
        self.assertIsNotNone(signal, "Strong bullish Chikou setup must emit ENTER_LONG signal")
        self.assertEqual(signal.action, "ENTER_LONG")
        self.assertEqual(signal.segment, "FUTURES")

    def test_builtin_strategies_registry(self):
        """Tests that CHIKOU_BREAKOUT is properly registered in BuiltinStrategies."""
        strategies = BuiltinStrategies.list_strategies()
        self.assertIn("CHIKOU_BREAKOUT", strategies)

        instance = BuiltinStrategies.create("CHIKOU_BREAKOUT")
        self.assertIsInstance(instance, ChikouBreakoutStrategy)

    def test_evaluator_auto_assigns_chikou_strategy(self):
        """Tests that enrich_plans_with_backtest_evaluations recognizes CHIKOU category."""
        df = self._generate_synthetic_df(n_bars=120, trend="bullish")
        data_dict = {"RELIANCE": df}
        plans = [{
            "symbol": "RELIANCE",
            "category": "CHIKOU_BREAKOUT",
            "trade_action": "BUY",
            "entry_price": float(df["close"].iloc[-1]),
            "stop_loss": float(df["close"].iloc[-1] * 0.96),
            "target_1": float(df["close"].iloc[-1] * 1.08),
        }]

        enriched = enrich_plans_with_backtest_evaluations(plans, data_dict=data_dict, cache_dict={})
        self.assertEqual(len(enriched), 1)
        self.assertIn("backtest_metrics", enriched[0])
        # Verify strategy label matches Chikou Span
        strat_lbl = enriched[0]["backtest_metrics"].get("strategy_label", "")
        self.assertIn("Chikou", strat_lbl)


if __name__ == "__main__":
    unittest.main()
