"""
Standardized Performance and Risk Analytics Engine.
Calculates institutional profitability metrics, risk parameters (Sharpe, Sortino, Max Drawdown),
cost drag decomposition, benchmark comparison, and monthly return matrices.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.backtesting.engine import Trade


@dataclass
class BacktestResult:
    """Comprehensive, auditable performance report produced by the backtest engine."""
    strategy_name: str
    strategy_version: str
    symbol: str
    start_date: str
    end_date: str
    starting_capital: float
    ending_capital: float
    gross_pnl: float
    net_pnl: float
    total_return_pct: float
    cagr_pct: float
    win_rate_pct: float
    profit_factor: float
    expectancy_per_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_trade: float
    avg_loss_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    # Risk
    max_drawdown_amount: float
    max_drawdown_pct: float
    max_drawdown_days: int
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    worst_trade_pnl: float
    worst_trade_pct: float
    # Costs
    total_costs: float
    total_brokerage: float
    total_stt: float
    total_exchange_charges: float
    total_gst: float
    total_stamp_duty: float
    total_slippage: float
    cost_drag_pct: float
    # Benchmark
    benchmark_name: str
    benchmark_return_pct: float
    alpha_pct: float
    # Time Series
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    trades: List[Trade] = field(default_factory=list)
    monthly_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_quality_report: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Provides an executive summary dictionary for UI and persistence."""
        return {
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "symbol": self.symbol,
            "date_range": f"{self.start_date} to {self.end_date}",
            "starting_capital": round(self.starting_capital, 2),
            "ending_capital": round(self.ending_capital, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "cagr_pct": round(self.cagr_pct, 2),
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "sortino_ratio": round(self.sortino_ratio, 2),
            "total_costs": round(self.total_costs, 2),
            "benchmark_return_pct": round(self.benchmark_return_pct, 2),
            "alpha_pct": round(self.alpha_pct, 2),
        }


class BacktestAnalytics:
    """
    Computes rigorous quantitative metrics from simulation outputs.
    Follows institutional performance presentation standards.
    """

    @classmethod
    def analyze(
        cls,
        engine_output: Dict[str, Any],
        benchmark_data: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.07,  # Standard 7% Indian G-Sec 10y baseline
        data_quality_report: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """Processes engine outputs into a complete BacktestResult."""
        equity_df = engine_output.get("equity_curve", pd.DataFrame())
        trades: List[Trade] = engine_output.get("trades", [])
        init_cap = float(engine_output.get("initial_capital", 200000.0))
        strat_name = str(engine_output.get("strategy_name", "Strategy"))
        strat_ver = str(engine_output.get("strategy_version", "1.0"))
        sym = str(engine_output.get("symbol", "NIFTY"))

        if equity_df.empty:
            return cls._empty_result(strat_name, strat_ver, sym, init_cap)

        equity_series = equity_df["equity"]
        start_cap = init_cap
        end_cap = float(equity_series.iloc[-1])
        start_dt = equity_df.index[0]
        end_dt = equity_df.index[-1]

        # 1. P&L & Return Metrics
        net_pnl = end_cap - start_cap
        tot_return_pct = (net_pnl / start_cap) * 100.0

        days_count = max(1, (end_dt - start_dt).days)
        years_count = max(0.01, days_count / 365.25)
        cagr_pct = ((end_cap / start_cap) ** (1.0 / years_count) - 1.0) * 100.0 if end_cap > 0 else -100.0

        # 2. Trade Breakdown
        n_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl < 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_trades * 100.0) if n_trades > 0 else 0.0

        gross_wins = sum(t.gross_pnl for t in wins)
        gross_losses = abs(sum(t.gross_pnl for t in losses))
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)

        avg_win = (sum(t.net_pnl for t in wins) / n_wins) if n_wins > 0 else 0.0
        avg_loss = (sum(t.net_pnl for t in losses) / n_losses) if n_losses > 0 else 0.0
        expectancy = (net_pnl / n_trades) if n_trades > 0 else 0.0

        # Consecutive streaks
        max_cw = 0
        curr_cw = 0
        max_cl = 0
        curr_cl = 0
        for t in trades:
            if t.net_pnl > 0:
                curr_cw += 1
                curr_cl = 0
                max_cw = max(max_cw, curr_cw)
            else:
                curr_cl += 1
                curr_cw = 0
                max_cl = max(max_cl, curr_cl)

        worst_trade_pnl = min((t.net_pnl for t in trades), default=0.0)
        worst_trade_pct = min((t.return_pct for t in trades), default=0.0) * 100.0

        # 3. Drawdown Series & Peak-to-Trough Metrics
        cummax = equity_series.cummax()
        drawdown_series = (equity_series - cummax) / cummax * 100.0
        max_dd_pct = abs(float(drawdown_series.min()))
        dd_amount_series = cummax - equity_series
        max_dd_amount = float(dd_amount_series.max())

        # Drawdown duration in days
        dd_days = 0
        cur_dd_days = 0
        for is_underwater in (equity_series < cummax):
            if is_underwater:
                cur_dd_days += 1
                dd_days = max(dd_days, cur_dd_days)
            else:
                cur_dd_days = 0

        # 4. Volatility, Sharpe, Sortino, Calmar
        daily_returns = equity_series.pct_change().dropna()
        ann_factor = 252.0 ** 0.5
        ann_vol_pct = float(daily_returns.std() * ann_factor * 100.0) if len(daily_returns) > 1 else 0.0

        rf_daily = risk_free_rate / 252.0
        excess_returns = daily_returns - rf_daily
        sharpe = float((excess_returns.mean() / daily_returns.std()) * ann_factor) if daily_returns.std() > 1e-6 else 0.0

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        sortino = float((excess_returns.mean() / downside_std) * ann_factor) if downside_std > 1e-6 else 0.0
        calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # 5. Cost Deconstruction
        tot_costs = sum(t.costs.total_cost for t in trades)
        tot_brokerage = sum(t.costs.brokerage for t in trades)
        tot_stt = sum(t.costs.stt for t in trades)
        tot_exchange = sum(t.costs.exchange_charges for t in trades)
        tot_gst = sum(t.costs.gst for t in trades)
        tot_stamp = sum(t.costs.stamp_duty for t in trades)
        tot_slippage = sum(t.costs.slippage for t in trades)
        gross_pnl = net_pnl + tot_costs
        cost_drag_pct = (tot_costs / abs(gross_pnl) * 100.0) if abs(gross_pnl) > 0 else 0.0

        # 6. Benchmark Comparison (Nifty 50 Buy & Hold)
        bench_ret_pct = 0.0
        bench_name = "NIFTY 50 Index"
        if benchmark_data is not None and not benchmark_data.empty:
            b_close = benchmark_data["close"].loc[equity_df.index[0]:equity_df.index[-1]]
            if len(b_close) >= 2:
                bench_ret_pct = ((b_close.iloc[-1] - b_close.iloc[0]) / b_close.iloc[0]) * 100.0
        elif "spot_close" in equity_df.columns:
            s_first = float(equity_df["spot_close"].iloc[0])
            s_last = float(equity_df["spot_close"].iloc[-1])
            bench_ret_pct = ((s_last - s_first) / s_first) * 100.0 if s_first > 0 else 0.0

        alpha_pct = tot_return_pct - bench_ret_pct

        # 7. Monthly Returns Heatmap Matrix
        monthly_df = cls._compute_monthly_returns(equity_series)

        return BacktestResult(
            strategy_name=strat_name,
            strategy_version=strat_ver,
            symbol=sym,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            starting_capital=start_cap,
            ending_capital=end_cap,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_return_pct=tot_return_pct,
            cagr_pct=cagr_pct,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            expectancy_per_trade=expectancy,
            total_trades=n_trades,
            winning_trades=n_wins,
            losing_trades=n_losses,
            avg_win_trade=avg_win,
            avg_loss_trade=avg_loss,
            max_consecutive_wins=max_cw,
            max_consecutive_losses=max_cl,
            max_drawdown_amount=max_dd_amount,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_days=dd_days,
            annualized_volatility_pct=ann_vol_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            worst_trade_pnl=worst_trade_pnl,
            worst_trade_pct=worst_trade_pct,
            total_costs=tot_costs,
            total_brokerage=tot_brokerage,
            total_stt=tot_stt,
            total_exchange_charges=tot_exchange,
            total_gst=tot_gst,
            total_stamp_duty=tot_stamp,
            total_slippage=tot_slippage,
            cost_drag_pct=cost_drag_pct,
            benchmark_name=bench_name,
            benchmark_return_pct=bench_ret_pct,
            alpha_pct=alpha_pct,
            equity_curve=equity_df,
            drawdown_series=drawdown_series,
            trades=trades,
            monthly_returns=monthly_df,
            data_quality_report=data_quality_report or {},
        )

    @staticmethod
    def _compute_monthly_returns(equity_series: pd.Series) -> pd.DataFrame:
        """Constructs Year x Month percentage return matrix."""
        try:
            m_resample = equity_series.resample("ME").last().pct_change() * 100.0
            if m_resample.empty:
                return pd.DataFrame()

            df_m = pd.DataFrame({
                "Year": m_resample.index.year,
                "Month": m_resample.index.strftime("%b"),
                "Return": m_resample.values
            })
            pivot = df_m.pivot(index="Year", columns="Month", values="Return")
            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            existing_cols = [m for m in month_order if m in pivot.columns]
            pivot = pivot[existing_cols]
            pivot["Annual"] = pivot.sum(axis=1)
            return pivot.round(2)
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _empty_result(strat_name: str, strat_ver: str, sym: str, capital: float) -> BacktestResult:
        return BacktestResult(
            strategy_name=strat_name,
            strategy_version=strat_ver,
            symbol=sym,
            start_date="",
            end_date="",
            starting_capital=capital,
            ending_capital=capital,
            gross_pnl=0.0,
            net_pnl=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            expectancy_per_trade=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win_trade=0.0,
            avg_loss_trade=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            max_drawdown_amount=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_days=0,
            annualized_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            worst_trade_pnl=0.0,
            worst_trade_pct=0.0,
            total_costs=0.0,
            total_brokerage=0.0,
            total_stt=0.0,
            total_exchange_charges=0.0,
            total_gst=0.0,
            total_stamp_duty=0.0,
            total_slippage=0.0,
            cost_drag_pct=0.0,
            benchmark_name="NIFTY 50",
            benchmark_return_pct=0.0,
            alpha_pct=0.0,
        )
