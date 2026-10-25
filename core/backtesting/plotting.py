"""
Plotly visualization helpers for Backtest Performance Reporting.
Generates institutional-grade Dark Terminal charts for equity curves,
underwater drawdown profiles, and statutory cost drag decomposition.
"""
from typing import Optional
import pandas as pd
import plotly.graph_objects as go

from core.backtesting.analytics import BacktestResult


def plot_backtest_equity_curve(df_equity: pd.DataFrame, strat_name: str, symbol: str) -> go.Figure:
    fig = go.Figure()
    if "spot_close" in df_equity.columns and not df_equity.empty:
        s0 = float(df_equity["spot_close"].iloc[0])
        init_cap = float(df_equity["equity"].iloc[0])
        bench_scaled = (df_equity["spot_close"] / s0) * init_cap
        fig.add_trace(go.Scatter(
            x=df_equity.index,
            y=bench_scaled,
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            name=f"{symbol} Buy & Hold (Benchmark)",
            hovertemplate="Benchmark: ₹%{y:,.0f}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=df_equity.index,
        y=df_equity["equity"],
        mode="lines",
        line=dict(color="#00E676", width=2.5),
        name=f"Strategy: {strat_name}",
        hovertemplate="Strategy Equity: ₹%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"📈 Historical Strategy Equity Curve vs Benchmark ({symbol})",
        paper_bgcolor="#131722",
        plot_bgcolor="#0e1117",
        font=dict(color="#E0E0E0", family="Inter, sans-serif"),
        xaxis=dict(title="Timeline", gridcolor="#1e222d"),
        yaxis=dict(title="Portfolio Equity (₹)", gridcolor="#1e222d"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_backtest_drawdown(drawdown_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index,
        y=drawdown_series.values,
        mode="lines",
        line=dict(color="#FF5252", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 82, 82, 0.2)",
        name="Drawdown %",
        hovertemplate="Drawdown: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        title="📉 Underwater Drawdown Profile (Peak-to-Trough)",
        paper_bgcolor="#131722",
        plot_bgcolor="#0e1117",
        font=dict(color="#E0E0E0", family="Inter, sans-serif"),
        xaxis=dict(title="Timeline", gridcolor="#1e222d"),
        yaxis=dict(title="Drawdown (%)", gridcolor="#1e222d"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=260,
        showlegend=False
    )
    return fig


def plot_cost_breakdown(result: BacktestResult) -> go.Figure:
    categories = ["Brokerage", "STT / CTT", "Exchange Txn", "GST (18%)", "Stamp Duty", "Slippage"]
    values = [
        result.total_brokerage,
        result.total_stt,
        result.total_exchange_charges,
        result.total_gst,
        result.total_stamp_duty,
        result.total_slippage
    ]
    colors = ["#38BDF8", "#F59E0B", "#A78BFA", "#EC4899", "#10B981", "#EF4444"]

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"₹{v:,.0f}" for v in values],
        textposition="auto",
        hovertemplate="%{x}: ₹%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="💸 Indian Statutory Taxes & Execution Slippage Drag Deconstruction",
        paper_bgcolor="#131722",
        plot_bgcolor="#0e1117",
        font=dict(color="#E0E0E0", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#1e222d"),
        yaxis=dict(title="Cost Incurred (₹)", gridcolor="#1e222d"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=280
    )
    return fig
