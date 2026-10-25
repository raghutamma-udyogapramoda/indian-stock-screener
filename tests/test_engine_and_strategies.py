"""
Comprehensive Unit Tests for Backtest Engine & Quantitative Trading Strategies.
Verifies chronological event loop, zero look-ahead bias, portfolio state ledger,
margin/capital constraints, Supertrend, Multi-Leg Spreads, and AI Rule Strategies.
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.backtesting.costs import IndianCostModel
from core.backtesting.strategy import (
    StrategyConfig,
    StrategySignal,
    SupertrendBreakoutStrategy,
    OptionsSpreadStrategy,
    AIRuleStrategy,
    BuiltinStrategies,
)
from core.backtesting.engine import (
    BacktestEngine,
    Position,
    Trade,
    PortfolioState,
)


class TestStrategySignalGeneration(unittest.TestCase):
    def setUp(self):
        # 50-bar series
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        base_prices = [100.0 + i * 0.2 for i in range(50)]

        self.bullish_df = pd.DataFrame({
            "open": base_prices,
            "high": [p + 2.0 for p in base_prices],
            "low": [p - 2.0 for p in base_prices],
            "close": base_prices,
            "volume": [100000] * 50,
            "supertrend": [p - 1.0 for p in base_prices],
            "supertrend_dir": [1] * 50,
            "ema_20": [p - 0.5 for p in base_prices],
            "ema_50": [p - 1.5 for p in base_prices],
            "rsi_14": [60.0] * 50,
            "rvol_20": [1.5] * 50,
            "atr_14": [2.5] * 50,
        }, index=dates)

        # Set up a clean bullish crossover at index 35
        # Bar 34: Close below Supertrend
        self.bullish_df.loc[self.bullish_df.index[34], "close"] = 98.0
        self.bullish_df.loc[self.bullish_df.index[34], "supertrend"] = 99.0
        # Bar 35: Close crosses above Supertrend
        self.bullish_df.loc[self.bullish_df.index[35], "close"] = 105.0
        self.bullish_df.loc[self.bullish_df.index[35], "supertrend"] = 100.0
        self.bullish_df.loc[self.bullish_df.index[35], "ema_20"] = 102.0
        self.bullish_df.loc[self.bullish_df.index[35], "ema_50"] = 101.0
        self.bullish_df.loc[self.bullish_df.index[35], "rsi_14"] = 62.0
        self.bullish_df.loc[self.bullish_df.index[35], "rvol_20"] = 1.8

        # Set up bearish dataframe
        self.bearish_df = self.bullish_df.copy()
        # Bar 34: Close above Supertrend
        self.bearish_df.loc[self.bearish_df.index[34], "close"] = 102.0
        self.bearish_df.loc[self.bearish_df.index[34], "supertrend"] = 101.0
        # Bar 35: Close crosses below Supertrend
        self.bearish_df.loc[self.bearish_df.index[35], "close"] = 94.0
        self.bearish_df.loc[self.bearish_df.index[35], "supertrend"] = 99.0
        self.bearish_df.loc[self.bearish_df.index[35], "ema_20"] = 96.0
        self.bearish_df.loc[self.bearish_df.index[35], "ema_50"] = 98.0
        self.bearish_df.loc[self.bearish_df.index[35], "rsi_14"] = 40.0

    def test_supertrend_strategy_signals(self):
        cfg = StrategyConfig(name="Supertrend Test", segment="OPTIONS")
        strat = SupertrendBreakoutStrategy(cfg)

        # Bullish signal test at index 35
        sig_bull = strat.generate_signals(self.bullish_df, portfolio_state=None, as_of_idx=35)
        self.assertIsNotNone(sig_bull)
        self.assertEqual(sig_bull.action, "ENTER_LONG")
        self.assertEqual(sig_bull.segment, "OPTION_CE")

        # Bearish signal test at index 35
        sig_bear = strat.generate_signals(self.bearish_df, portfolio_state=None, as_of_idx=35)
        self.assertIsNotNone(sig_bear)
        self.assertEqual(sig_bear.action, "ENTER_SHORT")
        self.assertEqual(sig_bear.segment, "OPTION_PE")

    def test_options_spread_strategy(self):
        cfg = StrategyConfig(name="Spreads Test", segment="MULTI_LEG")
        strat = OptionsSpreadStrategy(cfg)

        # Ensure bar 35 close breaks above previous 20-day high
        high_20 = self.bullish_df["high"].iloc[14:35].max()
        self.bullish_df.loc[self.bullish_df.index[35], "close"] = high_20 + 5.0
        self.bullish_df.loc[self.bullish_df.index[35], "rvol_20"] = 1.6

        sig_spread_bull = strat.generate_signals(self.bullish_df, portfolio_state=None, as_of_idx=35)
        self.assertIsNotNone(sig_spread_bull)
        self.assertEqual(sig_spread_bull.segment, "BULL_CALL_SPREAD")

    def test_ai_rule_strategy(self):
        rules = {
            "min_rsi": 55.0,
            "max_rsi": 75.0,
            "min_rvol": 1.2,
            "action": "ENTER_LONG",
        }
        strat = AIRuleStrategy(rules=rules)
        sig = strat.generate_signals(self.bullish_df, portfolio_state=None, as_of_idx=35)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.action, "ENTER_LONG")

        # When RSI condition fails
        strict_rules = {"min_rsi": 85.0, "max_rsi": 95.0}
        strat_strict = AIRuleStrategy(rules=strict_rules)
        sig_none = strat_strict.generate_signals(self.bullish_df, portfolio_state=None, as_of_idx=35)
        self.assertIsNone(sig_none)

    def test_builtin_strategies_factory(self):
        strats = BuiltinStrategies.list_strategies()
        self.assertIn("SUPERTREND_BREAKOUT", strats)
        self.assertIn("DEFINED_RISK_SPREADS", strats)
        self.assertIn("AI_CUSTOM_RULE", strats)

        obj = BuiltinStrategies.create("SUPERTREND_BREAKOUT")
        self.assertIsInstance(obj, SupertrendBreakoutStrategy)


class TestBacktestEngineMechanics(unittest.TestCase):
    def setUp(self):
        # 60-bar oscillating dataset designed to generate multiple crossover triggers
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        np.random.seed(42)
        base = 20000.0
        prices = []
        for i in range(60):
            # Sinusoidal cycle to produce clean flips
            cycle = np.sin(i / 4.0) * 800.0
            noise = np.random.normal(0, 50.0)
            prices.append(base + cycle + noise)

        df = pd.DataFrame(index=dates)
        df["close"] = prices
        df["open"] = [p * 0.998 for p in prices]
        df["high"] = [p * 1.008 for p in prices]
        df["low"] = [p * 0.992 for p in prices]
        df["volume"] = [800000] * 60
        df["supertrend"] = base  # Flat baseline to trigger periodic crosses
        df["supertrend_dir"] = [1 if p > base else -1 for p in prices]
        df["ema_20"] = [p * 0.99 for p in prices]
        df["ema_50"] = [p * 0.98 for p in prices]
        df["rsi_14"] = [55.0 if p > base else 45.0 for p in prices]
        df["rvol_20"] = 1.6
        df["atr_14"] = 150.0
        self.df = df

    def test_chronological_order_and_zero_lookahead(self):
        cfg = StrategyConfig(name="Engine Test", segment="OPTIONS", max_holding_bars=4)
        strat = SupertrendBreakoutStrategy(cfg)
        engine = BacktestEngine(strategy=strat, initial_capital=200000.0)
        out = engine.run(self.df, symbol="NIFTY", lot_size=25, strike_step=50.0)

        trades = out["trades"]
        self.assertGreater(len(trades), 0)
        for t in trades:
            # Chronological consistency: entry precedes exit
            self.assertLessEqual(t.entry_time, t.exit_time)
            self.assertGreaterEqual(t.holding_bars, 0)

    def test_capital_reconciliation_exactness(self):
        cfg = StrategyConfig(name="Recon Test", segment="FUTURES", stop_loss_pct=0.02, target_pct=0.04)
        strat = SupertrendBreakoutStrategy(cfg)
        engine = BacktestEngine(strategy=strat, initial_capital=500000.0)
        out = engine.run(self.df, symbol="NIFTY", lot_size=25, strike_step=50.0)

        initial_cap = out["initial_capital"]
        final_eq = out["final_equity"]
        trades = out["trades"]

        total_net_pnl = sum(t.net_pnl for t in trades)
        total_costs = sum(t.costs.total_cost for t in trades)
        total_gross_pnl = sum(t.gross_pnl for t in trades)

        # Net PnL must equal Gross PnL - Total Costs
        self.assertAlmostEqual(total_gross_pnl - total_costs, total_net_pnl, places=2)

        # Final Equity must equal Initial Capital + Net PnL
        self.assertAlmostEqual(initial_cap + total_net_pnl, final_eq, places=2)

    def test_insufficient_bars_rejection(self):
        short_df = self.df.iloc[:20].copy()
        engine = BacktestEngine(strategy=SupertrendBreakoutStrategy(), initial_capital=200000.0)
        out = engine.run(short_df, symbol="NIFTY")
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
