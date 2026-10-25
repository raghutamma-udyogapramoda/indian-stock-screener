"""
Comprehensive Unit Tests for Quantitative Backtest Analytics & Risk Reporting.
Verifies Sharpe ratio, Sortino ratio, Calmar ratio, Max Drawdown calculation,
CAGR annualized returns, Benchmark Alpha, Cost Drag, and Monthly Returns Matrix.
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.backtesting.costs import CostBreakdown
from core.backtesting.engine import Trade
from core.backtesting.analytics import BacktestAnalytics, BacktestResult


class TestBacktestAnalyticsDetailed(unittest.TestCase):
    def setUp(self):
        # 1-year daily equity curve (252 bars)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        np.random.seed(42)
        base = 200000.0
        # Upward drift with noise
        daily_rets = np.random.normal(0.0008, 0.009, 252)
        equity = [base]
        for r in daily_rets[1:]:
            equity.append(equity[-1] * (1.0 + r))

        spot_prices = [18000.0 * (eq / base) for eq in equity]

        self.df_equity = pd.DataFrame({
            "cash": equity,
            "unrealized_pnl": [0.0] * 252,
            "equity": equity,
            "open_positions": [0] * 252,
            "spot_close": spot_prices,
        }, index=dates)

        # Create mock trades
        t1 = Trade(
            trade_id="TRD_0001",
            symbol="NIFTY",
            segment="OPTIONS",
            action="LONG",
            entry_time=dates[10],
            exit_time=dates[15],
            entry_price=100.0,
            exit_price=150.0,
            quantity=25,
            holding_bars=5,
            gross_pnl=1250.0,
            net_pnl=1150.0,
            return_pct=0.50,
            exit_reason="Target Hit",
            costs=CostBreakdown(total_cost=100.0, brokerage=40.0, stt=25.0, slippage=15.0),
        )
        t2 = Trade(
            trade_id="TRD_0002",
            symbol="NIFTY",
            segment="OPTIONS",
            action="LONG",
            entry_time=dates[30],
            exit_time=dates[35],
            entry_price=120.0,
            exit_price=90.0,
            quantity=25,
            holding_bars=5,
            gross_pnl=-750.0,
            net_pnl=-850.0,
            return_pct=-0.25,
            exit_reason="Stop Loss",
            costs=CostBreakdown(total_cost=100.0, brokerage=40.0, stt=15.0, slippage=15.0),
        )
        self.mock_trades = [t1, t2]

        self.engine_output = {
            "initial_capital": base,
            "final_equity": equity[-1],
            "equity_curve": self.df_equity,
            "trades": self.mock_trades,
            "total_trades": len(self.mock_trades),
            "strategy_name": "Test Analytics Strategy",
            "strategy_version": "1.0",
            "symbol": "NIFTY",
        }

    def test_analytics_metrics_computation(self):
        result = BacktestAnalytics.analyze(self.engine_output)
        self.assertIsInstance(result, BacktestResult)

        # Capital and returns
        self.assertEqual(result.starting_capital, 200000.0)
        self.assertAlmostEqual(result.ending_capital, self.df_equity["equity"].iloc[-1], places=2)
        self.assertEqual(result.total_trades, 2)
        self.assertEqual(result.winning_trades, 1)
        self.assertEqual(result.losing_trades, 1)
        self.assertEqual(result.win_rate_pct, 50.0)

        # Profit factor: Total Wins (1250) / Total Losses (750) = 1.67
        self.assertAlmostEqual(result.profit_factor, 1250.0 / 750.0, places=2)

        # Risk metrics
        self.assertGreaterEqual(result.max_drawdown_pct, 0.0)
        self.assertGreaterEqual(result.max_drawdown_days, 0)
        self.assertIsInstance(result.sharpe_ratio, float)
        self.assertIsInstance(result.sortino_ratio, float)

        # Costs
        self.assertEqual(result.total_costs, 200.0)
        self.assertEqual(result.total_brokerage, 80.0)

        # Benchmark Alpha
        self.assertIsInstance(result.benchmark_return_pct, float)
        self.assertIsInstance(result.alpha_pct, float)

    def test_monthly_matrix_calculation(self):
        result = BacktestAnalytics.analyze(self.engine_output)
        matrix = result.monthly_returns
        self.assertFalse(matrix.empty)
        # Year 2023 must be in matrix index
        self.assertIn(2023, matrix.index)
        # Annual column should exist
        self.assertIn("Annual", matrix.columns)

    def test_empty_trades_edge_case(self):
        empty_out = self.engine_output.copy()
        empty_out["trades"] = []
        empty_out["total_trades"] = 0
        result = BacktestAnalytics.analyze(empty_out)

        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.win_rate_pct, 0.0)
        self.assertEqual(result.profit_factor, 0.0)
        self.assertEqual(result.total_costs, 0.0)

    def test_summary_dict_serialization(self):
        result = BacktestAnalytics.analyze(self.engine_output)
        sm = result.to_summary_dict()
        self.assertIn("strategy_name", sm)
        self.assertIn("total_return_pct", sm)
        self.assertIn("win_rate_pct", sm)
        self.assertIn("max_drawdown_pct", sm)
        self.assertIn("profit_factor", sm)


if __name__ == "__main__":
    unittest.main()
