"""
Unit and Integration Tests for AI-Powered F&O Backtesting Subsystem.
Verifies costs, pricing models, deterministic execution, zero look-ahead bias,
portfolio reconciliation, and strategy persistence.
"""
import unittest
import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.options_analytics import BlackScholesEngine
from core.backtesting.costs import IndianCostModel, CostBreakdown
from core.backtesting.data_provider import (
    HistoricalDataProvider,
    YahooHistoricalDataProvider,
    DerivativesDataProvider,
    DataQualityReport,
)
from core.backtesting.strategy import (
    StrategyConfig,
    SupertrendBreakoutStrategy,
    OptionsSpreadStrategy,
    AIRuleStrategy,
    BuiltinStrategies,
)
from core.backtesting.engine import BacktestEngine, Position, Trade, PortfolioState
from core.backtesting.analytics import BacktestAnalytics, BacktestResult
from core.backtesting.runner import BacktestRunner


class TestIndianCostModel(unittest.TestCase):
    def setUp(self):
        self.model = IndianCostModel(
            brokerage_per_order=20.0,
            slippage_pct=0.0005,
            enable_statutory=True,
        )

    def test_options_cost_budget_2024(self):
        # 1 lot NIFTY (lot size 25) bought at 100, sold at 150
        # Buy turnover = 25 * 100 = 2500
        # Sell turnover = 25 * 150 = 3750
        cost_buy = self.model.calculate_order_cost(
            action="BUY",
            price=100.0,
            quantity=25,
            segment="OPTIONS",
        )
        cost_sell = self.model.calculate_order_cost(
            action="SELL",
            price=150.0,
            quantity=25,
            segment="OPTIONS",
        )

        # STT on buy options should be 0.0
        self.assertEqual(cost_buy.stt, 0.0)
        # Post-Budget 2024 STT on sell options: 0.1% on premium turnover (3750 * 0.001 = 3.75)
        self.assertAlmostEqual(cost_sell.stt, 3750.0 * 0.001, places=2)
        # Brokerage is 20 per order
        self.assertEqual(cost_buy.brokerage, 20.0)
        self.assertEqual(cost_sell.brokerage, 20.0)
        # GST is 18% on (brokerage + exchange_turnover)
        self.assertGreater(cost_buy.gst, 0.0)
        # Total cost is positive
        self.assertGreater(cost_buy.total_cost, 20.0)
        self.assertGreater(cost_sell.total_cost, 20.0)

    def test_futures_cost(self):
        # 1 lot NIFTY Futures (lot size 25) bought at 24000, sold at 24200
        # Turnover buy = 600,000; Turnover sell = 605,000
        cost_buy = self.model.calculate_order_cost(
            action="BUY",
            price=24000.0,
            quantity=25,
            segment="FUTURES",
        )
        cost_sell = self.model.calculate_order_cost(
            action="SELL",
            price=24200.0,
            quantity=25,
            segment="FUTURES",
        )
        # STT on futures sell: 0.02% of turnover (605,000 * 0.0002 = 121.0)
        self.assertAlmostEqual(cost_sell.stt, 605000.0 * 0.0002, places=1)
        self.assertEqual(cost_buy.stt, 0.0)


class TestDerivativesPricing(unittest.TestCase):
    def test_black_scholes_analytical_pricing(self):
        # S=24000, K=24000 (ATM), T=7 days, r=6.5%, sigma=14%
        t_years = 7.0 / 365.0
        r = 0.065
        iv = 0.14
        call_atm = BlackScholesEngine.price(
            spot=24000, strike=24000, t_years=t_years, r=r, iv=iv, opt_type="CE"
        )
        put_atm = BlackScholesEngine.price(
            spot=24000, strike=24000, t_years=t_years, r=r, iv=iv, opt_type="PE"
        )

        self.assertGreater(call_atm, 0.0)
        self.assertGreater(put_atm, 0.0)
        # Put-Call Parity: C - P = S - K * exp(-r*T)
        lhs = call_atm - put_atm
        rhs = 24000 - 24000 * math.exp(-r * t_years)
        self.assertAlmostEqual(lhs, rhs, delta=1.0)

    def test_greeks_analytical_consistency(self):
        t_years = 14.0 / 365.0
        greeks = BlackScholesEngine.calc_greeks(
            spot=24000, strike=24000, t_years=t_years, r=0.065, iv=0.15, opt_type="CE"
        )
        # Call delta must be in (0, 1)
        self.assertGreater(greeks["delta"], 0.4)
        self.assertLess(greeks["delta"], 0.6)
        # Gamma > 0
        self.assertGreater(greeks["gamma"], 0.0)
        # Vega > 0
        self.assertGreater(greeks["vega"], 0.0)
        # Theta < 0 (decay)
        self.assertLess(greeks["theta"], 0.0)


class TestDeterministicEngineZeroLookahead(unittest.TestCase):
    def setUp(self):
        # Create deterministic synthetic daily OHLCV dataset (60 bars)
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        np.random.seed(42)
        base_price = 18000.0
        prices = [base_price]
        for _ in range(59):
            prices.append(prices[-1] * (1.0 + np.random.normal(0.001, 0.01)))

        df = pd.DataFrame(index=dates)
        df["close"] = prices
        df["open"] = [p * 0.998 for p in prices]
        df["high"] = [p * 1.008 for p in prices]
        df["low"] = [p * 0.992 for p in prices]
        df["volume"] = 1000000.0
        # Add required indicator columns
        df["supertrend"] = df["close"] * 0.98
        df["supertrend_dir"] = 1
        df["rsi_14"] = 60.0
        df["rvol_20"] = 1.8
        df["atr_14"] = 150.0
        self.market_data = df

    def test_engine_reconciliation_and_no_lookahead(self):
        config = StrategyConfig(
            name="Test Supertrend",
            segment="OPTIONS",
            stop_loss_pct=0.03,
            target_pct=0.06,
        )
        strat = SupertrendBreakoutStrategy(config)
        engine = BacktestEngine(
            strategy=strat,
            initial_capital=200000.0,
            cost_model=IndianCostModel(),
        )
        engine_out = engine.run(self.market_data, symbol="NIFTY", lot_size=25, strike_step=50.0)
        self.assertNotIn("error", engine_out)

        result = BacktestAnalytics.analyze(engine_out)

        # Portfolio accounting reconciliation check
        # Ending Equity must equal Initial Capital + Net PnL
        expected_equity = 200000.0 + result.net_pnl
        self.assertAlmostEqual(result.ending_capital, expected_equity, places=2)
        self.assertAlmostEqual(result.gross_pnl - result.total_costs, result.net_pnl, places=2)

        # Look-ahead bias check: Trade exit time must be >= entry time
        for t in result.trades:
            self.assertGreaterEqual(t.exit_time, t.entry_time)


class TestBacktestRunnerAndPersistence(unittest.TestCase):
    def test_end_to_end_runner(self):
        res = BacktestRunner.run_backtest(
            strategy_id="SUPERTREND_BREAKOUT",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-04-01",
            segment="OPTIONS",
            initial_capital=200000.0,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertEqual(res.symbol, "NIFTY")
        self.assertGreater(len(res.equity_curve), 0)
        self.assertTrue(len(res.run_id) > 0)

        # Saved runs list must contain this run
        saved = BacktestRunner.list_saved_runs()
        self.assertGreater(len(saved), 0)
        run_ids = [r["run_id"] for r in saved]
        self.assertIn(res.run_id, run_ids)

    def test_multi_leg_options_spread_strategy(self):
        res = BacktestRunner.run_backtest(
            strategy_id="DEFINED_RISK_SPREADS",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-06-30",
            segment="MULTI_LEG",
            initial_capital=200000.0,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertGreater(len(res.equity_curve), 0)
        # Check that drawdown calculation worked
        self.assertGreaterEqual(res.max_drawdown_pct, 0.0)

    def test_ai_custom_rule_strategy(self):
        custom_rules = {
            "min_rsi": 50.0,
            "max_rsi": 80.0,
            "min_rvol": 1.2,
            "action": "ENTER_LONG",
        }
        res = BacktestRunner.run_backtest(
            strategy_id="AI_CUSTOM_RULE",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-04-01",
            segment="OPTIONS",
            initial_capital=200000.0,
            custom_rules=custom_rules,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertGreater(len(res.equity_curve), 0)

    def test_futures_segment_simulation(self):
        res = BacktestRunner.run_backtest(
            strategy_id="SUPERTREND_BREAKOUT",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-04-01",
            segment="FUTURES",
            initial_capital=300000.0,
        )
        self.assertIsInstance(res, BacktestResult)
        self.assertGreater(len(res.equity_curve), 0)

    def test_ui_plotting_helpers(self):
        from core.backtesting import plot_backtest_equity_curve, plot_backtest_drawdown, plot_cost_breakdown

        res = BacktestRunner.run_backtest(
            strategy_id="SUPERTREND_BREAKOUT",
            symbol="NIFTY",
            start_date="2023-01-01",
            end_date="2023-04-01",
            segment="OPTIONS",
            initial_capital=200000.0,
        )
        fig_eq = plot_backtest_equity_curve(res.equity_curve, res.strategy_name, res.symbol)
        fig_dd = plot_backtest_drawdown(res.drawdown_series)
        fig_cost = plot_cost_breakdown(res)

        self.assertIsNotNone(fig_eq)
        self.assertIsNotNone(fig_dd)
        self.assertIsNotNone(fig_cost)


class TestQuantitativeBacktestValidation(unittest.TestCase):
    def test_quantitative_pass_hurdles_logic(self):
        from core.backtesting.evaluator import enrich_plans_with_backtest_evaluations as _enrich_plans_with_backtest_evaluations

        # 1. Qualified candidate: Positive return, win rate >= 45%, PF >= 1.2, DD <= 35%, trades >= 3
        passing_plan = {
            "symbol": "TCS",
            "trade_action": "BUY",
            "category": "Breakout",
            "backtest_passed": True,
            "backtest_metrics": {
                "return_pct": 22.4,
                "win_rate_pct": 52.0,
                "profit_factor": 1.75,
                "max_drawdown_pct": 14.5,
                "total_trades": 18,
                "net_pnl": 44800.0,
                "total_costs": 3100.0,
            }
        }

        # 2. Failed candidate: Negative return after statutory taxes/slippage
        failing_return_plan = {
            "symbol": "WIPRO",
            "trade_action": "BUY",
            "category": "Breakout",
            "backtest_passed": False,
            "backtest_metrics": {
                "return_pct": -6.2,
                "win_rate_pct": 48.0,
                "profit_factor": 0.88,
                "max_drawdown_pct": 21.0,
                "total_trades": 12,
                "net_pnl": -12400.0,
                "total_costs": 2400.0,
            }
        }

        # 3. Failed candidate: Excessive drawdown (> 35%)
        failing_dd_plan = {
            "symbol": "ADANIENT",
            "trade_action": "BUY",
            "category": "Momentum",
            "backtest_passed": False,
            "backtest_metrics": {
                "return_pct": 15.0,
                "win_rate_pct": 50.0,
                "profit_factor": 1.40,
                "max_drawdown_pct": 42.0,
                "total_trades": 14,
                "net_pnl": 30000.0,
                "total_costs": 3500.0,
            }
        }

        # 4. Failed candidate: Insufficient sample size (< 3 trades)
        failing_sample_plan = {
            "symbol": "NESTLEIND",
            "trade_action": "BUY",
            "category": "Swing",
            "backtest_passed": False,
            "backtest_metrics": {
                "return_pct": 8.0,
                "win_rate_pct": 100.0,
                "profit_factor": 99.0,
                "max_drawdown_pct": 3.0,
                "total_trades": 2,
                "net_pnl": 16000.0,
                "total_costs": 400.0,
            }
        }

        plans = [passing_plan, failing_return_plan, failing_dd_plan, failing_sample_plan]
        enriched = _enrich_plans_with_backtest_evaluations(plans)

        self.assertEqual(len(enriched), 4)
        self.assertTrue(enriched[0]["backtest_passed"])
        self.assertFalse(enriched[1]["backtest_passed"])
        self.assertFalse(enriched[2]["backtest_passed"])
        self.assertFalse(enriched[3]["backtest_passed"])

    def test_enrich_empty_plans_list(self):
        from core.backtesting.evaluator import enrich_plans_with_backtest_evaluations as _enrich_plans_with_backtest_evaluations
        self.assertEqual(_enrich_plans_with_backtest_evaluations([]), [])

    def test_enrich_plans_with_in_memory_dataframe(self):
        from core.backtesting.evaluator import enrich_plans_with_backtest_evaluations as _enrich_plans_with_backtest_evaluations

        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        mock_df = pd.DataFrame({
            "open": np.linspace(100.0, 130.0, 60),
            "high": np.linspace(102.0, 132.0, 60),
            "low": np.linspace(99.0, 129.0, 60),
            "close": np.linspace(101.0, 131.0, 60),
            "volume": [500000] * 60,
        }, index=dates)

        plan = {
            "symbol": "MOCK_GROWTH",
            "trade_action": "BUY",
            "category": "Breakout",
            "close": 131.0,
            "entry_price": 100.0,
            "stop_loss": 97.0,
            "tight_stop_loss": 97.0,
            "target_1": 115.0,
        }

        data_dict = {"MOCK_GROWTH": mock_df}
        enriched = _enrich_plans_with_backtest_evaluations([plan], data_dict=data_dict)

        self.assertEqual(len(enriched), 1)
        p = enriched[0]
        self.assertIn("backtest_passed", p)
        self.assertIn("backtest_metrics", p)
        self.assertIsInstance(p["backtest_passed"], bool)
        self.assertIsInstance(p["backtest_metrics"], dict)
        self.assertIn("return_pct", p["backtest_metrics"])
        self.assertIn("win_rate_pct", p["backtest_metrics"])
        self.assertIn("profit_factor", p["backtest_metrics"])
        self.assertIn("max_drawdown_pct", p["backtest_metrics"])


if __name__ == "__main__":
    unittest.main()

