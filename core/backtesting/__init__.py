"""
Backtesting and Strategy Research Subsystem.
Supports historical simulation of Equities, Index/Stock Futures, and Multi-Leg Options.
"""

from core.backtesting.costs import IndianCostModel, CostBreakdown
from core.backtesting.data_provider import (
    HistoricalDataProvider,
    YahooHistoricalDataProvider,
    LocalHistoricalDataProvider,
    DerivativesDataProvider,
    DataQualityReport,
)
from core.backtesting.strategy import Strategy, StrategySignal, StrategyConfig, BuiltinStrategies
from core.backtesting.engine import BacktestEngine, Position, Trade, PortfolioState
from core.backtesting.analytics import BacktestAnalytics, BacktestResult
from core.backtesting.runner import BacktestRunner
from core.backtesting.plotting import (
    plot_backtest_equity_curve,
    plot_backtest_drawdown,
    plot_cost_breakdown,
)

__all__ = [
    "IndianCostModel",
    "CostBreakdown",
    "HistoricalDataProvider",
    "YahooHistoricalDataProvider",
    "LocalHistoricalDataProvider",
    "DerivativesDataProvider",
    "DataQualityReport",
    "Strategy",
    "StrategySignal",
    "StrategyConfig",
    "BuiltinStrategies",
    "BacktestEngine",
    "Position",
    "Trade",
    "PortfolioState",
    "BacktestAnalytics",
    "BacktestResult",
    "BacktestRunner",
    "plot_backtest_equity_curve",
    "plot_backtest_drawdown",
    "plot_cost_breakdown",
]
