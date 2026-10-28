"""
Unit tests for Corporate Results & Earnings Scanner (core/results_scanner.py).
Tests calendar filtering, technical next-day move predictions, and scorecard integrity.
"""

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from core.results_scanner import ResultsCalendarManager, ResultsChartPredictor


class TestResultsScanner(unittest.TestCase):
    def setUp(self):
        # Generate 40 synthetic trading sessions for testing technical next-day derivations
        dates = pd.date_range("2024-08-01", periods=40, freq="B")
        np.random.seed(42)
        base_price = 2500.0
        closes = [base_price]
        for _ in range(39):
            closes.append(closes[-1] * (1.0 + np.random.normal(0.002, 0.015)))

        df = pd.DataFrame(index=dates)
        df["close"] = closes
        df["open"] = [c * (1.0 + np.random.normal(0.0, 0.005)) for c in closes]
        df["high"] = [max(o, c) * 1.01 for o, c in zip(df["open"], df["close"])]
        df["low"] = [min(o, c) * 0.99 for o, c in zip(df["open"], df["close"])]
        df["volume"] = [1000000 + np.random.randint(-100000, 200000) for _ in closes]

        # Populate core indicators
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["sma_200"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["rsi_14"] = 55.0
        df["atr_14"] = (df["high"] - df["low"]).rolling(14, min_periods=1).mean()
        df["rvol_20"] = 1.35
        df["clv"] = 0.65
        df["bb_bandwidth"] = 0.09
        self.sample_df = df

    def test_calendar_categories(self):
        """Tests that calendar provides declared, upcoming this week, and upcoming next week stocks."""
        all_catalog = ResultsCalendarManager.get_all_results_catalog()
        self.assertGreater(len(all_catalog), 10)

        declared = ResultsCalendarManager.get_declared_stocks()
        self.assertGreater(len(declared), 4)
        for s in declared:
            self.assertEqual(s["status"], "DECLARED")
            self.assertIn("revenue_cr", s)
            self.assertIn("net_profit_cr", s)

        this_week = ResultsCalendarManager.get_upcoming_this_week()
        self.assertGreater(len(this_week), 2)
        for s in this_week:
            self.assertEqual(s["status"], "UPCOMING")
            self.assertEqual(s["timing_window"], "THIS_WEEK")

        next_week = ResultsCalendarManager.get_upcoming_next_week()
        self.assertGreater(len(next_week), 2)
        for s in next_week:
            self.assertEqual(s["status"], "UPCOMING")
            self.assertEqual(s["timing_window"], "NEXT_WEEK")

    def test_stock_details_lookup(self):
        """Tests lookup for curated and unlisted symbols."""
        rel = ResultsCalendarManager.get_stock_result_details("RELIANCE")
        self.assertIsNotNone(rel)
        self.assertEqual(rel["symbol"], "RELIANCE")
        self.assertEqual(rel["verdict"], "IN_LINE")

        maz = ResultsCalendarManager.get_stock_result_details("MAZDOCK")
        self.assertIsNotNone(maz)
        self.assertEqual(maz["symbol"], "MAZDOCK")
        self.assertEqual(maz["verdict"], "BEAT")

        unlisted = ResultsCalendarManager.get_stock_result_details("CUSTOMXYZ")
        self.assertIsNotNone(unlisted)
        self.assertEqual(unlisted["symbol"], "CUSTOMXYZ")

    def test_technical_next_day_prediction(self):
        """Tests next-day technical trajectory, expected ranges, and action trigger derivation."""
        meta = ResultsCalendarManager.get_stock_result_details("RELIANCE")
        pred = ResultsChartPredictor.analyze_next_day_chart_move(self.sample_df, meta)

        self.assertIn("current_price", pred)
        self.assertGreater(pred["current_price"], 0)
        self.assertGreater(pred["expected_move_pts"], 0)
        self.assertGreater(pred["upper_band"], pred["current_price"])
        self.assertLess(pred["lower_band"], pred["current_price"])
        self.assertGreater(pred["buy_trigger"], 0)
        self.assertGreater(pred["target_1"], pred["buy_trigger"])
        self.assertLess(pred["sell_trigger"], pred["buy_trigger"])
        self.assertIn("tactical_notes", pred)
        self.assertGreater(len(pred["tactical_notes"]), 0)

    def test_results_scanner_ui_rendering(self):
        """Tests that Recent Results Scanner mounts with stock name cards and without exceptions."""
        from streamlit.testing.v1 import AppTest
        from pathlib import Path
        app_path = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")
        at = AppTest.from_file(app_path, default_timeout=60)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "trader_results"
        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed mounting Results Scanner: {at.exception}")

        # Verify stock cards with names only
        btn_labels = [b.label for b in at.button]
        self.assertTrue(any("**RELIANCE**" in b for b in btn_labels), "Expected **RELIANCE** card button")
        self.assertTrue(any("**TCS**" in b for b in btn_labels), "Expected **TCS** card button")
        self.assertTrue(any("**MAZDOCK**" in b for b in btn_labels), "Expected **MAZDOCK** card button")


if __name__ == "__main__":
    unittest.main()
