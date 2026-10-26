"""
Quantitative Validation and Backtest Evaluation Engine for AI Trade Plans.
Implements industry-standard 5-pillar statistical hurdles:
1. Net Total Return > 0.0% (profitable after all Indian statutory taxes and slippage)
2. Win Rate >= 45% (or >= 38% with asymmetric Profit Factor >= 1.25x)
3. Profit Factor >= 1.20x (Gross Profit / Gross Loss)
4. Max Drawdown <= 35% (protects against account drawdown and margin calls)
5. Total Trades >= 3 (minimum statistical significance)
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from core.backtesting.strategy import BuiltinStrategies
from core.backtesting.runner import BacktestRunner
from core.universe import UniverseManager


def evaluate_backtest_hurdles(
    return_pct: float,
    win_rate_pct: float,
    profit_factor: float,
    max_drawdown_pct: float,
    total_trades: int
) -> bool:
    """
    Evaluates backtest metrics against industry-standard quantitative hurdles.
    """
    return (
        return_pct > 0.0
        and (win_rate_pct >= 45.0 or (profit_factor >= 1.25 and win_rate_pct >= 38.0))
        and profit_factor >= 1.20
        and max_drawdown_pct <= 35.0
        and total_trades >= 3
    )


def enrich_plans_with_backtest_evaluations(
    plans: List[Dict[str, Any]],
    data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    cache_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Backtests each plan against 1 year of historical data using realistic statutory trading costs
    and evaluates each against industry-standard quantitative hurdles.
    Results are cached in cache_dict (or Streamlit session state if available)
    so subsequent tab interactions run at 0ms latency.
    """
    if not plans:
        return []

    # Check session state or custom cache dict
    cache = cache_dict
    if cache is None:
        try:
            import streamlit as st
            if hasattr(st, "session_state"):
                cache = st.session_state
        except Exception:
            cache = None

    cache_key = f"cached_plans_bt_eval_{len(plans)}_{'_'.join(str(p.get('symbol', '')) for p in plans[:6])}"
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    enriched = []
    strat_dict = BuiltinStrategies.list_strategies()

    for p in plans:
        p_copy = dict(p)

        # Allow pre-computed backtest metrics (e.g. from tests or fixtures)
        if "backtest_passed" in p_copy and "backtest_metrics" in p_copy:
            enriched.append(p_copy)
            continue

        sym = p_copy.get("symbol", "")
        raw_cat = str(p_copy.get("category", "")).upper()
        act = str(p_copy.get("trade_action", p_copy.get("action", ""))).upper()

        # 1. Strategy Model Selection
        if "CHIKOU" in raw_cat:
            default_strat_id = "CHIKOU_BREAKOUT"
        elif "DIP" in raw_cat:
            default_strat_id = "RSI_PULLBACK"
        elif "BTST" in raw_cat:
            default_strat_id = "BTST_MOMENTUM"
        elif "SWING" in raw_cat or "SPREAD" in raw_cat:
            default_strat_id = "DEFINED_RISK_SPREADS"
        elif "BREAKDOWN" in raw_cat or ("SELL" in act and "PE" not in act):
            default_strat_id = "EMA_CROSSOVER"
        else:
            default_strat_id = "SUPERTREND_BREAKOUT"

        # 2. Segment
        if "PE" in act or "CE" in act or "OPTION" in raw_cat:
            default_seg = "OPTIONS"
        elif "SPREAD" in raw_cat:
            default_seg = "MULTI_LEG"
        elif UniverseManager.is_commodity(sym) or UniverseManager.is_global_commodity(sym) or "FUT" in raw_cat:
            default_seg = "FUTURES"
        else:
            default_seg = "EQUITY"

        # 3. Parameters
        try:
            entry_val = float(p_copy.get("entry_price", p_copy.get("close", 100.0)))
            sl_val = float(p_copy.get("tight_stop_loss", p_copy.get("stop_loss", entry_val * 0.97)))
            tgt_val = float(p_copy.get("target_1", entry_val * 1.06))
            sl_pct = max(0.015, min(0.05, abs(entry_val - sl_val) / max(1.0, entry_val)))
            target_pct = max(0.03, min(0.12, abs(tgt_val - entry_val) / max(1.0, entry_val)))
        except Exception:
            sl_pct = 0.025
            target_pct = 0.06

        holding_bars = min(30, max(5, int(p_copy.get("holding_period_days", 15))))
        in_df = data_dict.get(sym) if isinstance(data_dict, dict) else None

        # 4. Execute Backtest
        try:
            bt_res = BacktestRunner.run_backtest(
                strategy_id=default_strat_id,
                symbol=sym,
                start_date="2023-01-01",
                end_date="2024-01-01",
                segment=default_seg,
                initial_capital=200000.0,
                stop_loss_pct=sl_pct,
                target_pct=target_pct,
                max_holding_bars=holding_bars,
                enable_statutory_costs=True,
                historical_data=in_df,
            )
        except Exception:
            bt_res = None

        if bt_res and bt_res.total_trades > 0:
            ret = float(bt_res.total_return_pct)
            wr = float(bt_res.win_rate_pct)
            pf = float(bt_res.profit_factor)
            dd = float(bt_res.max_drawdown_pct)
            trades = int(bt_res.total_trades)

            is_pass = evaluate_backtest_hurdles(
                return_pct=ret,
                win_rate_pct=wr,
                profit_factor=pf,
                max_drawdown_pct=dd,
                total_trades=trades,
            )

            p_copy["backtest_passed"] = is_pass
            p_copy["backtest_result"] = bt_res
            p_copy["backtest_metrics"] = {
                "return_pct": ret,
                "win_rate_pct": wr,
                "profit_factor": pf,
                "max_drawdown_pct": dd,
                "total_trades": trades,
                "net_pnl": bt_res.net_pnl,
                "total_costs": bt_res.total_costs,
                "strategy_id": default_strat_id,
                "strategy_label": strat_dict.get(default_strat_id, default_strat_id),
                "segment": default_seg,
            }
        else:
            p_copy["backtest_passed"] = False
            p_copy["backtest_result"] = None
            p_copy["backtest_metrics"] = {
                "return_pct": 0.0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "total_trades": 0,
                "net_pnl": 0.0,
                "total_costs": 0.0,
                "strategy_id": default_strat_id,
                "strategy_label": strat_dict.get(default_strat_id, default_strat_id),
                "segment": default_seg,
            }

        enriched.append(p_copy)

    if cache is not None:
        try:
            cache[cache_key] = enriched
        except Exception:
            pass

    return enriched
