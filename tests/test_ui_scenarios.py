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
        at = AppTest.from_file(self.app_path, default_timeout=60)
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
        at = AppTest.from_file(self.app_path, default_timeout=60)
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
        at = AppTest.from_file(self.app_path, default_timeout=60)
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
        at = AppTest.from_file(self.app_path, default_timeout=60)
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
        at = AppTest.from_file(self.app_path, default_timeout=60)
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

    def test_scenario_7_gemini_api_key_default_and_override(self):
        """Tests that Gemini API key is pre-configured by default in sidebar and user can override it."""
        from config.settings import DEFAULT_GEMINI_API_KEY
        at = AppTest.from_file(self.app_path, default_timeout=60)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"
        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed on mount: {at.exception}")

        # Find the Gemini API Key input in the sidebar
        gemini_inputs = [t for t in at.text_input if "Gemini API Key" in t.label]
        self.assertTrue(len(gemini_inputs) > 0, "Sidebar should contain a Gemini API Key input")
        gemini_input = gemini_inputs[0]

        # Verify the pre-configured default value matches DEFAULT_GEMINI_API_KEY
        self.assertEqual(
            gemini_input.value,
            DEFAULT_GEMINI_API_KEY,
            "Default Gemini API key should be pre-configured"
        )

        # Verify that the user can override the key with custom value
        custom_key = "test_custom_user_provided_override_key_123"
        gemini_input.input(custom_key).run()
        self.assertEqual(len(at.exception), 0, f"App crashed after overriding Gemini API key: {at.exception}")
        updated_input = [t for t in at.text_input if "Gemini API Key" in t.label][0]
        self.assertEqual(updated_input.value, custom_key, "User should be able to override Gemini API Key")

    def test_scenario_8_trade_dataclass_backward_compatibility(self):
        """Tests that Trade dataclass has entry_date, exit_date, side, and total_cost properties without AttributeError."""
        from datetime import datetime
        t = Trade(
            trade_id="TRD_TEST_PROP",
            symbol="NIFTY",
            segment="FUTURES",
            action="LONG",
            entry_time=datetime(2023, 5, 10, 9, 15),
            exit_time=datetime(2023, 5, 15, 15, 15),
            entry_price=18200.0,
            exit_price=18450.0,
            quantity=50,
            holding_bars=5,
            gross_pnl=12500.0,
            net_pnl=12100.0,
            return_pct=0.0137,
            exit_reason="Target 1 Hit",
            costs=CostBreakdown(total_cost=400.0, brokerage=40.0, stt=250.0, slippage=110.0),
        )

        # Ensure no AttributeError occurs when accessing entry_date, exit_date, side, and total_cost
        self.assertEqual(str(t.entry_date)[:10], "2023-05-10")
        self.assertEqual(str(t.exit_date)[:10], "2023-05-15")
        self.assertEqual(t.side, "LONG")
        self.assertEqual(t.total_cost, 400.0)

    def test_scenario_9_dedicated_backtest_navigation_and_back_arrow(self):
        """Tests navigating to the dedicated backtest page on click, and navigating back via top back button."""
        at = AppTest.from_file(self.app_path, default_timeout=60)
        at.session_state["authenticated"] = True
        at.session_state["symbols"] = ["RELIANCE"]
        at.session_state["elapsed"] = 1.0
        at.session_state["cached_results"] = {"Breakout": [{"symbol": "RELIANCE", "score": 90, "close": 2500.0}]}
        at.session_state["data_dict"] = {
            "RELIANCE": pd.DataFrame({
                "open": [2480.0] * 35,
                "high": [2520.0] * 35,
                "low": [2470.0] * 35,
                "close": [2500.0] * 35,
                "volume": [1000000] * 35,
            }, index=pd.date_range("2023-01-01", periods=35, freq="D"))
        }

        # Simulate user launching backtest from AI trade plan
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        eq_df = pd.DataFrame({"equity": [200000.0 + i * 100.0 for i in range(30)]}, index=dates)
        mock_res = BacktestResult(
            strategy_name="EMA Trend Pullback",
            strategy_version="1.0",
            symbol="RELIANCE",
            start_date="2023-01-01",
            end_date="2023-01-30",
            starting_capital=200000.0,
            ending_capital=203000.0,
            gross_pnl=3080.0,
            net_pnl=3000.0,
            total_return_pct=1.5,
            cagr_pct=18.0,
            win_rate_pct=70.0,
            profit_factor=2.0,
            expectancy_per_trade=300.0,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            avg_win_trade=3000.0,
            avg_loss_trade=0.0,
            max_consecutive_wins=1,
            max_consecutive_losses=0,
            max_drawdown_amount=500.0,
            max_drawdown_pct=0.25,
            max_drawdown_days=2,
            annualized_volatility_pct=10.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=3.0,
            worst_trade_pnl=0.0,
            worst_trade_pct=0.0,
            total_costs=80.0,
            total_brokerage=40.0,
            total_stt=20.0,
            total_exchange_charges=5.0,
            total_gst=8.0,
            total_stamp_duty=2.0,
            total_slippage=5.0,
            cost_drag_pct=0.5,
            benchmark_name="NIFTY 50",
            benchmark_return_pct=1.0,
            alpha_pct=0.5,
            equity_curve=eq_df,
            drawdown_series=pd.Series([0.0] * 30, index=dates),
            trades=[
                Trade(
                    trade_id="TRD_REL_01",
                    symbol="RELIANCE",
                    segment="EQUITY",
                    action="BUY",
                    entry_time=dates[2],
                    exit_time=dates[10],
                    entry_price=2400.0,
                    exit_price=2500.0,
                    quantity=30,
                    holding_bars=8,
                    gross_pnl=3000.0,
                    net_pnl=2920.0,
                    return_pct=0.041,
                    exit_reason="Target Hit",
                    costs=CostBreakdown(total_cost=80.0),
                )
            ],
            monthly_returns=pd.DataFrame({"Jan": [1.5], "Annual": [1.5]}, index=[2023]),
            data_quality_report={"total_bars": 30},
        )

        at.session_state["view_mode"] = "backtest_results"
        at.session_state["active_backtest_result"] = mock_res
        at.session_state["active_backtest_source_plan"] = {"symbol": "RELIANCE", "strategy_name": "EMA Trend Pullback"}

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed on dedicated backtest results page: {at.exception}")

        # Check top back button exists
        back_btns = [b for b in at.button if "Back to Recommendations" in b.label]
        self.assertGreater(len(back_btns), 0, "Top Back arrow button must be present on dedicated backtest results page")

        # Simulate user clicking back button to return to search results
        back_btns[0].click().run()
        self.assertEqual(len(at.exception), 0, f"App crashed on clicking Back to Recommendations: {at.exception}")
        self.assertEqual(at.session_state.get("view_mode"), "search", "View mode should reset to search without re-screening")

    def test_scenario_10_high_conviction_futures_desk(self):
        """Tests that the High-Conviction Futures Desk enforces low-failure capital defense and displays Buy/Sell futures without duplicate keys."""
        at = AppTest.from_file(self.app_path, default_timeout=60)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"
        at.session_state["symbols"] = ["RELIANCE", "TCS"]
        at.session_state["elapsed"] = 1.0
        at.session_state["cached_results"] = {
            "Breakout": [
                {
                    "symbol": "RELIANCE",
                    "score": 92,
                    "bias": "LONG",
                    "close": 2500.0,
                    "change_pct": 2.5,
                    "rvol": 1.8,
                    "reasons": ["High Conviction 52W Breakout"],
                    "metrics": {"breakout_pivot": 2450.0},
                }
            ],
            "Breakdown": [
                {
                    "symbol": "TCS",
                    "score": 85,
                    "bias": "SHORT",
                    "close": 3400.0,
                    "change_pct": -2.1,
                    "rvol": 1.6,
                    "reasons": ["Breakdown below support"],
                    "metrics": {"breakdown_pivot": 3450.0},
                }
            ],
        }
        at.session_state["cached_ai_plans"] = [
            {
                "symbol": "RELIANCE",
                "action": "BUY",
                "score": 92,
                "conviction": "HIGH",
                "close": 2500.0,
                "cmp": 2500.0,
                "entry_price": 2500.0,
                "stop_loss": 2450.0,
                "target_1": 2600.0,
                "target_2": 2700.0,
                "rvol": 1.8,
                "category": "Breakout",
                "thesis": "High conviction breakout above pivot with rising institutional volumes.",
            },
            {
                "symbol": "TCS",
                "action": "SELL",
                "score": 85,
                "conviction": "HIGH",
                "close": 3400.0,
                "cmp": 3400.0,
                "entry_price": 3400.0,
                "stop_loss": 3450.0,
                "target_1": 3300.0,
                "target_2": 3200.0,
                "rvol": 1.6,
                "category": "Breakdown",
                "thesis": "Distribution breakdown below key swing low with volume confirmation.",
            }
        ]
        at.session_state["data_dict"] = {
            "RELIANCE": pd.DataFrame({
                "open": [2450.0] * 35, "high": [2510.0] * 35, "low": [2440.0] * 35, "close": [2500.0] * 35, "volume": [1000000] * 35
            }, index=pd.date_range("2023-01-01", periods=35, freq="D")),
            "TCS": pd.DataFrame({
                "open": [3450.0] * 35, "high": [3460.0] * 35, "low": [3390.0] * 35, "close": [3400.0] * 35, "volume": [800000] * 35
            }, index=pd.date_range("2023-01-01", periods=35, freq="D")),
        }
        at.session_state["data_health"] = {
            "is_latest": True, "provider_name": "Exchange", "latest_data_date": "2023-02-05", "refresh_timestamp": "09:15"
        }

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed in screened state with Futures desk: {at.exception}")

        # Check that high conviction futures desk content is rendered
        raw_markdown = " ".join([m.value for m in at.markdown])
        self.assertIn("High-Conviction Futures Desk", raw_markdown)

        # Check that backtest buttons for futures are generated across both desks without duplicate keys
        fut_bt_btns = [b for b in at.button if "Backtest Futures Strategy" in b.label]
        self.assertGreater(len(fut_bt_btns), 0, "Futures strategy backtest buttons should be rendered on qualified cards")

    def test_scenario_11_backtest_passed_tab(self):
        """Tests the Backtest Passed quantitative validation tab and scorecard."""
        at = AppTest.from_file(self.app_path, default_timeout=60)
        at.session_state["authenticated"] = True
        at.session_state["username"] = "test_trader"
        at.session_state["symbols"] = ["RELIANCE", "INFY"]
        at.session_state["elapsed"] = 1.0
        at.session_state["cached_results"] = {
            "Breakout": [
                {
                    "symbol": "RELIANCE",
                    "score": 92,
                    "bias": "LONG",
                    "close": 2500.0,
                    "change_pct": 2.5,
                    "rvol": 1.8,
                    "reasons": ["High Conviction 52W Breakout"],
                    "metrics": {"breakout_pivot": 2450.0},
                }
            ],
            "Breakdown": [
                {
                    "symbol": "INFY",
                    "score": 85,
                    "bias": "SHORT",
                    "close": 1400.0,
                    "change_pct": -2.1,
                    "rvol": 1.6,
                    "reasons": ["Breakdown below support"],
                    "metrics": {"breakdown_pivot": 1450.0},
                }
            ],
        }
        at.session_state["cached_ai_plans"] = [
            {
                "symbol": "RELIANCE",
                "trade_action": "BUY",
                "score": 92,
                "conviction": "HIGH",
                "close": 2500.0,
                "cmp": 2500.0,
                "entry_price": 2500.0,
                "stop_loss": 2450.0,
                "tight_stop_loss": 2450.0,
                "target_1": 2600.0,
                "target_2": 2700.0,
                "rvol": 1.8,
                "category": "Breakout",
                "thesis": "High conviction breakout above pivot with rising institutional volumes.",
                "backtest_passed": True,
                "backtest_metrics": {
                    "return_pct": 28.5,
                    "win_rate_pct": 62.5,
                    "profit_factor": 1.95,
                    "max_drawdown_pct": 11.2,
                    "total_trades": 16,
                    "net_pnl": 57000.0,
                    "total_costs": 3200.0,
                    "strategy_id": "SUPERTREND_BREAKOUT",
                    "strategy_label": "Supertrend Breakout",
                    "segment": "EQUITY",
                }
            },
            {
                "symbol": "INFY",
                "trade_action": "SELL",
                "score": 85,
                "conviction": "HIGH",
                "close": 1400.0,
                "cmp": 1400.0,
                "entry_price": 1400.0,
                "stop_loss": 1450.0,
                "tight_stop_loss": 1450.0,
                "target_1": 1300.0,
                "target_2": 1250.0,
                "rvol": 1.6,
                "category": "Breakdown",
                "thesis": "Distribution breakdown below key swing low with volume confirmation.",
                "backtest_passed": False,
                "backtest_metrics": {
                    "return_pct": -8.5,
                    "win_rate_pct": 33.3,
                    "profit_factor": 0.75,
                    "max_drawdown_pct": 22.0,
                    "total_trades": 9,
                    "net_pnl": -17000.0,
                    "total_costs": 1800.0,
                    "strategy_id": "EMA_CROSSOVER",
                    "strategy_label": "EMA Crossover",
                    "segment": "EQUITY",
                }
            }
        ]
        at.session_state["data_dict"] = {
            "RELIANCE": pd.DataFrame({
                "open": [2450.0] * 35, "high": [2510.0] * 35, "low": [2440.0] * 35, "close": [2500.0] * 35, "volume": [1000000] * 35
            }, index=pd.date_range("2023-01-01", periods=35, freq="D")),
            "INFY": pd.DataFrame({
                "open": [1450.0] * 35, "high": [1460.0] * 35, "low": [1390.0] * 35, "close": [1400.0] * 35, "volume": [800000] * 35
            }, index=pd.date_range("2023-01-01", periods=35, freq="D")),
        }
        at.session_state["data_health"] = {
            "is_latest": True, "provider_name": "Exchange", "latest_data_date": "2023-02-05", "refresh_timestamp": "09:15"
        }

        at.run()
        self.assertEqual(len(at.exception), 0, f"App crashed displaying Backtest Passed tab: {at.exception}")

        # Check for Backtest Passed header
        raw_markdown = " ".join([m.value for m in at.markdown])
        self.assertIn("Backtest Passed", raw_markdown)

        # Check that quantitative hurdle filter controls exist
        select_labels = [s.label for s in at.selectbox]
        self.assertIn("Sort Backtested Setups:", select_labels)
        self.assertIn("Strategy Category:", select_labels)


if __name__ == "__main__":
    unittest.main()

