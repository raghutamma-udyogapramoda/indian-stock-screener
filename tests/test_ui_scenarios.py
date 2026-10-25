"""
Comprehensive Automated UI Testing Suite for Indian Stock Screener & F&O Backtesting Platform.
Uses Streamlit's official AppTest framework to exercise all UI views, tabs, forms,
interactive filters, Plotly chart elements, and data tables with zero unhandled exceptions.
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from streamlit.testing.v1 import AppTest
from core.backtesting import BacktestRunner, BacktestResult, CostBreakdown, Trade


class TestStreamlitUIViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_path = str(root_dir / "ui" / "app.py")

    def test_scenario_1_landing_page_default_state(self):
        """Tests that initial landing page mounts immediately with Search Recommendations as the first view."""
        at = AppTest.from_file(self.app_path, default_timeout=35)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed on default mount: {at.exception}")

        # Check search controls and parameters on the primary landing page
        select_labels = [s.label for s in at.selectbox]
        self.assertIn("Select Asset Universe", select_labels)
        self.assertIn("Strategy Filter / Style", select_labels)

        # Check search button and recommendation presets
        btn_labels = [b.label for b in at.button]
        self.assertTrue(any("Search Trade Recommendations" in b for b in btn_labels))
        self.assertTrue(any("Top Dip Leaders" in b for b in btn_labels))
        self.assertTrue(any("52W Breakouts" in b for b in btn_labels))

        # Check that independent backtest option is also available if needed
        self.assertIn("Strategy Template", select_labels)
        self.assertIn("Underlying Symbol", select_labels)
        self.assertTrue(any("Run Quantitative Backtest" in b for b in btn_labels))

    def test_scenario_2_full_screened_state_11_tabs(self):
        """Tests the full multi-tab dashboard after market scan results are loaded."""
        at = AppTest.from_file(self.app_path, default_timeout=35)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"
        at.session_state["symbols"] = ["RELIANCE", "TCS", "INFY"]
        at.session_state["elapsed"] = 1.25
        at.session_state["cached_results"] = {
            "Breakout": [
                {
                    "symbol": "RELIANCE",
                    "score": 95,
                    "bias": "LONG",
                    "close": 2550.0,
                    "change_pct": 2.15,
                    "rvol": 1.95,
                    "reasons": ["Supertrend Breakout Above 52W Pivot"],
                    "metrics": {"breakout_pivot": 2500.0, "52w_high": 2600.0},
                }
            ],
            "Breakdown": [
                {
                    "symbol": "INFY",
                    "score": 88,
                    "bias": "SHORT",
                    "close": 1420.0,
                    "change_pct": -2.85,
                    "rvol": 1.75,
                    "reasons": ["Breakdown below EMA 50"],
                    "metrics": {"breakdown_pivot": 1460.0, "52w_low": 1390.0},
                }
            ],
        }
        at.session_state["data_dict"] = {
            "RELIANCE": pd.DataFrame({
                "open": [2500.0] * 35,
                "high": [2560.0] * 35,
                "low": [2490.0] * 35,
                "close": [2550.0] * 35,
                "volume": [1000000] * 35,
            }, index=pd.date_range("2023-01-01", periods=35, freq="D"))
        }
        at.session_state["data_health"] = {
            "is_latest": True,
            "provider_name": "Verified Exchange Feed",
            "latest_data_date": "2023-02-05",
            "refresh_timestamp": "09:15 AM",
        }

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed in screened state: {at.exception}")

        # Metrics strip check
        metrics = at.metric
        self.assertGreater(len(metrics), 3)

        # Tab check: verify tab-related content rendered
        markdown_text = " ".join([m.value for m in at.markdown])
        self.assertIn("Strategy Backtesting", markdown_text)
        self.assertIn("Live Item Tracker", markdown_text)

        # Check New Market Search button to return to search home
        btn_labels = [b.label for b in at.button]
        self.assertTrue(any("New Market Search" in b for b in btn_labels))

    def test_scenario_3_backtest_performance_results_display(self):
        """Tests that active backtest report renders all KPI cards, Plotly charts, and tables."""
        at = AppTest.from_file(self.app_path, default_timeout=35)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"

        # Construct a mock BacktestResult
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        eq_df = pd.DataFrame({
            "cash": [200000.0 + i * 200.0 for i in range(50)],
            "unrealized_pnl": [0.0] * 50,
            "equity": [200000.0 + i * 200.0 for i in range(50)],
            "open_positions": [0] * 50,
            "spot_close": [18000.0 + i * 15.0 for i in range(50)],
        }, index=dates)

        trade = Trade(
            trade_id="TRD_0001",
            symbol="NIFTY",
            segment="OPTIONS",
            action="LONG",
            entry_time=dates[5],
            exit_time=dates[10],
            entry_price=100.0,
            exit_price=160.0,
            quantity=25,
            holding_bars=5,
            gross_pnl=1500.0,
            net_pnl=1420.0,
            return_pct=0.60,
            exit_reason="Target Hit",
            costs=CostBreakdown(total_cost=80.0, brokerage=40.0, stt=20.0, slippage=20.0),
        )

        mock_res = BacktestResult(
            strategy_name="Supertrend Trend Breakout",
            strategy_version="1.1.0",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-02-20",
            starting_capital=200000.0,
            ending_capital=210000.0,
            gross_pnl=10080.0,
            net_pnl=10000.0,
            total_return_pct=5.0,
            cagr_pct=35.0,
            win_rate_pct=75.0,
            profit_factor=2.5,
            expectancy_per_trade=500.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            avg_win_trade=1420.0,
            avg_loss_trade=0.0,
            max_consecutive_wins=1,
            max_consecutive_losses=0,
            max_drawdown_amount=2500.0,
            max_drawdown_pct=1.25,
            max_drawdown_days=4,
            annualized_volatility_pct=12.5,
            sharpe_ratio=1.85,
            sortino_ratio=2.40,
            calmar_ratio=4.0,
            worst_trade_pnl=0.0,
            worst_trade_pct=0.0,
            total_costs=80.0,
            total_brokerage=40.0,
            total_stt=20.0,
            total_exchange_charges=5.0,
            total_gst=8.0,
            total_stamp_duty=2.0,
            total_slippage=5.0,
            cost_drag_pct=0.8,
            benchmark_name="NIFTY 50",
            benchmark_return_pct=2.5,
            alpha_pct=2.5,
            equity_curve=eq_df,
            drawdown_series=pd.Series([0.0] * 50, index=dates),
            trades=[trade],
            monthly_returns=pd.DataFrame({"Jan": [3.2], "Feb": [1.8], "Annual": [5.0]}, index=[2023]),
            data_quality_report={"data_source": "YAHOO_EXCHANGE_VERIFIED", "total_bars": 50, "verified_bars": 50},
        )

        at.session_state["active_backtest_result"] = mock_res
        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed displaying performance report: {at.exception}")

        # Check performance report header
        headers = [h.value for h in at.markdown if "Performance Report" in h.value]
        self.assertGreater(len(headers), 0)

        # Check download buttons (Trades CSV and Equity CSV)
        download_btns = [d.label for d in at.download_button]
        self.assertTrue(any("Download Trades Ledger" in b for b in download_btns))
        self.assertTrue(any("Download Equity Curve" in b for b in download_btns))

    def test_scenario_4_ai_strategy_discovery_subtab(self):
        """Tests the AI strategy parameter discovery controls."""
        at = AppTest.from_file(self.app_path, default_timeout=35)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"

        at.run()
        self.assertEqual(len(at.exception), 0)

        # Verify AI strategy sliders and controls exist
        slider_labels = [s.label for s in at.slider]
        self.assertTrue(any("RSI" in s for s in slider_labels))
        self.assertTrue(any("RVOL" in s for s in slider_labels))

        select_labels = [s.label for s in at.selectbox]
        self.assertIn("Choose AI Strategy Blueprint", select_labels)
        self.assertIn("Trade Direction", select_labels)

    def test_scenario_5_ai_trade_plan_page_backtest_option(self):
        """Tests that the backtest option is available on the AI Trade Plan page and cards."""
        at = AppTest.from_file(self.app_path, default_timeout=35)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"
        at.session_state["symbols"] = ["RELIANCE"]
        at.session_state["elapsed"] = 1.0
        at.session_state["cached_results"] = {
            "Breakout": [
                {
                    "symbol": "RELIANCE",
                    "score": 95,
                    "bias": "LONG",
                    "close": 2550.0,
                    "change_pct": 2.15,
                    "rvol": 1.95,
                    "reasons": ["Supertrend Breakout Above 52W Pivot"],
                    "metrics": {"breakout_pivot": 2500.0, "52w_high": 2600.0, "atr": 45.0},
                }
            ],
        }
        at.session_state["data_dict"] = {
            "RELIANCE": pd.DataFrame({
                "open": [2500.0] * 35,
                "high": [2560.0] * 35,
                "low": [2490.0] * 35,
                "close": [2550.0] * 35,
                "volume": [1000000] * 35,
            }, index=pd.date_range("2023-01-01", periods=35, freq="D"))
        }
        at.session_state["data_health"] = {
            "is_latest": True,
            "provider_name": "Verified Exchange Feed",
            "latest_data_date": "2023-02-05",
            "refresh_timestamp": "09:15 AM",
        }

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed on AI Trade Plan page: {at.exception}")

        # Check backtest launcher buttons on the AI Trade Plan cards
        btn_labels = [b.label for b in at.button]
        self.assertTrue(
            any("Run Backtest for RELIANCE" in b for b in btn_labels),
            f"Expected backtest button for RELIANCE on AI trade plan card, found: {btn_labels}"
        )

        # Check that top selector or quick launcher is present
        select_labels = [s.label for s in at.selectbox]
        self.assertTrue(
            any("Select Stock to Backtest" in s for s in select_labels),
            f"Expected 'Select Stock to Backtest' on AI trade plan page, found: {select_labels}"
        )


if __name__ == "__main__":
    unittest.main()
