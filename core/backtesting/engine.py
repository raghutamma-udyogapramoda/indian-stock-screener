"""
Deterministic Event-Driven Chronological Backtesting Engine.
Executes bar-by-bar historical simulations without look-ahead bias,
supporting Equities, Futures, Single-Leg Options, and Multi-Leg Options Spreads.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.backtesting.costs import IndianCostModel, CostBreakdown
from core.backtesting.data_provider import DerivativesDataProvider, HistoricalDataProvider
from core.backtesting.strategy import Strategy, StrategySignal
from core.options_analytics import BlackScholesEngine


@dataclass
class Position:
    """Active open market position (single or multi-leg)."""
    pos_id: str
    symbol: str
    segment: str  # "EQUITY", "FUTURES", "OPTION_CE", "OPTION_PE", "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD"
    entry_time: datetime
    entry_bar_idx: int
    entry_price: float
    quantity: int
    is_long: bool = True
    strike: float = 0.0
    option_type: str = ""
    legs: List[Dict[str, Any]] = field(default_factory=list)
    stop_loss: float = 0.0
    target: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    accumulated_costs: CostBreakdown = field(default_factory=CostBreakdown)
    thesis: str = ""

    @property
    def capital_outlay(self) -> float:
        """Returns the initial capital or margin outlay tied up in this position."""
        if "SPREAD" in self.segment:
            return sum(l["price"] * l["qty"] for l in self.legs if l["action"] == "BUY")
        elif self.segment.startswith("OPTION"):
            return self.entry_price * self.quantity
        elif self.segment == "FUTURES":
            return self.entry_price * self.quantity * 0.15
        else:
            return self.entry_price * self.quantity

    def update_valuation(self, current_spot: float, current_opt_price: Optional[float] = None):
        """Updates position mark-to-market based on current price."""
        if self.segment.startswith("OPTION"):
            p_cur = current_opt_price if current_opt_price is not None else self.entry_price
            self.current_price = p_cur
            mult = 1.0 if self.is_long else -1.0
            self.unrealized_pnl = (p_cur - self.entry_price) * self.quantity * mult
            self.unrealized_pnl_pct = ((p_cur - self.entry_price) / max(0.01, self.entry_price)) * mult

        elif "SPREAD" in self.segment:
            # Multi-leg spread valuation
            tot_pnl = 0.0
            for leg in self.legs:
                act = leg["action"]
                ot = leg["type"]
                k = leg["strike"]
                p_in = leg["price"]
                q = leg["qty"]
                # Approximate intrinsic + time value
                cur_leg_p = max(0.05, (current_spot - k) if ot == "CE" else (k - current_spot))
                leg_pnl = (cur_leg_p - p_in) * q if act == "BUY" else (p_in - cur_leg_p) * q
                tot_pnl += leg_pnl
            self.unrealized_pnl = tot_pnl
            cost_basis = max(100.0, sum(l["price"] * l["qty"] for l in self.legs if l["action"] == "BUY"))
            self.unrealized_pnl_pct = tot_pnl / cost_basis

        else:
            # Equity or Futures
            p_cur = current_spot
            self.current_price = p_cur
            mult = 1.0 if self.is_long else -1.0
            self.unrealized_pnl = (p_cur - self.entry_price) * self.quantity * mult
            self.unrealized_pnl_pct = ((p_cur - self.entry_price) / max(0.01, self.entry_price)) * mult


@dataclass
class Trade:
    """Historical completed trade record for the ledger."""
    trade_id: str
    symbol: str
    segment: str
    action: str  # "LONG", "SHORT", "SPREAD"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    holding_bars: int
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    costs: CostBreakdown
    legs_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "segment": self.segment,
            "action": self.action,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M") if hasattr(self.entry_time, "strftime") else str(self.entry_time),
            "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M") if hasattr(self.exit_time, "strftime") else str(self.exit_time),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "quantity": self.quantity,
            "holding_bars": self.holding_bars,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct * 100.0, 2),
            "exit_reason": self.exit_reason,
            "total_costs": round(self.costs.total_cost, 2),
            "brokerage": round(self.costs.brokerage, 2),
            "stt": round(self.costs.stt, 2),
            "exchange_charges": round(self.costs.exchange_charges, 2),
            "gst": round(self.costs.gst, 2),
            "stamp_duty": round(self.costs.stamp_duty, 2),
            "slippage": round(self.costs.slippage, 2),
            "legs_summary": self.legs_summary,
        }

    @property
    def entry_date(self) -> str:
        return self.entry_time.strftime("%Y-%m-%d") if hasattr(self.entry_time, "strftime") else str(self.entry_time)[:10]

    @property
    def exit_date(self) -> str:
        return self.exit_time.strftime("%Y-%m-%d") if hasattr(self.exit_time, "strftime") else str(self.exit_time)[:10]

    @property
    def side(self) -> str:
        return self.action

    @property
    def total_cost(self) -> float:
        return self.costs.total_cost if self.costs else 0.0


@dataclass
class PortfolioState:
    """Ledger state at any discrete bar during backtest."""
    timestamp: datetime
    cash: float
    starting_capital: float
    open_positions: List[Position] = field(default_factory=list)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_costs_paid: float = 0.0
    margin_utilized: float = 0.0
    equity: float = 0.0

    def update_equity(self):
        self.unrealized_pnl = sum(p.unrealized_pnl for p in self.open_positions)
        self.equity = self.cash + self.unrealized_pnl


class BacktestEngine:
    """
    Core chronological simulation engine.
    Ensures zero look-ahead bias:
      1. Signals generated at bar t close are filled at bar t+1 executable price.
      2. Orders are subject to realistic execution slippage and statutory costs.
      3. Precise portfolio reconciliation.
    """

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 200000.0,
        cost_model: Optional[IndianCostModel] = None,
        risk_per_trade_pct: float = 0.02,
        max_active_positions: int = 3,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_model = cost_model or IndianCostModel()
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_active_positions = max_active_positions
        self.derivatives_provider = DerivativesDataProvider()

    def run(
        self,
        market_data: pd.DataFrame,
        symbol: str = "NIFTY",
        lot_size: int = 50,
        strike_step: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Executes chronological backtest over market_data series.
        """
        if market_data.empty or len(market_data) < 35:
            return {"error": "Insufficient market bars to execute backtest (minimum 35 required)"}

        # Set symbol metadata on dataframe
        market_data = market_data.copy()
        setattr(market_data, "symbol", symbol)

        cash = self.initial_capital
        open_positions: List[Position] = []
        closed_trades: List[Trade] = []
        equity_records: List[Dict[str, Any]] = []
        trade_counter = 0

        pending_signal: Optional[StrategySignal] = None
        pending_signal_bar: int = -1

        timestamps = market_data.index
        n_bars = len(market_data)

        for i in range(n_bars):
            curr_bar = market_data.iloc[i]
            curr_dt = timestamps[i]
            spot_close = float(curr_bar["close"])
            spot_open = float(curr_bar["open"])

            # -------------------------------------------------------------
            # STEP 1: Process Pending Signals from Bar t-1 (Execution at Open)
            # -------------------------------------------------------------
            if pending_signal is not None and len(open_positions) < self.max_active_positions:
                sig = pending_signal
                exec_price = spot_open  # Realistic fill at next bar's open price
                seg = sig.segment

                # Option Strike Calculation
                atm_k = round(exec_price / strike_step) * strike_step
                chosen_strike = atm_k
                if sig.strike_selection == "ITM_1":
                    chosen_strike = atm_k - strike_step if "CE" in seg else atm_k + strike_step
                elif sig.strike_selection == "OTM_1":
                    chosen_strike = atm_k + strike_step if "CE" in seg else atm_k - strike_step

                trade_counter += 1
                pos_id = f"POS_{trade_counter:04d}"

                if seg in ("OPTION_CE", "OPTION_PE"):
                    opt_t = "CE" if "CE" in seg else "PE"
                    # Analytical pricing at entry (7-day horizon standard)
                    opt_prem = max(1.0, round(BlackScholesEngine.price(exec_price, chosen_strike, 7.0/365.0, opt_type=opt_t), 2))
                    qty = lot_size
                    cost_entry = self.cost_model.calculate_order_cost("BUY", opt_prem, qty, segment="OPTIONS")

                    total_outlay = (opt_prem * qty) + cost_entry.total_cost
                    if cash >= total_outlay:
                        cash -= total_outlay
                        pos = Position(
                            pos_id=pos_id,
                            symbol=symbol,
                            segment=seg,
                            entry_time=curr_dt,
                            entry_bar_idx=i,
                            entry_price=opt_prem,
                            quantity=qty,
                            is_long=True,
                            strike=chosen_strike,
                            option_type=opt_t,
                            stop_loss=sig.stop_loss_pct,
                            target=sig.target_pct,
                            current_price=opt_prem,
                            accumulated_costs=cost_entry,
                            thesis=sig.thesis,
                        )
                        open_positions.append(pos)

                elif "SPREAD" in seg:
                    # Multi-leg Spread (Bull Call / Bear Put)
                    qty = lot_size
                    is_bull = "BULL" in seg
                    k1 = atm_k
                    k2 = atm_k + strike_step if is_bull else atm_k - strike_step
                    t_leg = "CE" if is_bull else "PE"

                    p1 = max(1.0, round(BlackScholesEngine.price(exec_price, k1, 7.0/365.0, opt_type=t_leg), 2))
                    p2 = max(0.5, round(BlackScholesEngine.price(exec_price, k2, 7.0/365.0, opt_type=t_leg), 2))

                    legs = [
                        {"action": "BUY", "type": t_leg, "strike": k1, "price": p1, "qty": qty},
                        {"action": "SELL", "type": t_leg, "strike": k2, "price": p2, "qty": qty},
                    ]
                    cost1 = self.cost_model.calculate_order_cost("BUY", p1, qty, segment="OPTIONS")
                    cost2 = self.cost_model.calculate_order_cost("SELL", p2, qty, segment="OPTIONS")
                    tot_spread_cost = cost1 + cost2

                    net_debit = (p1 - p2) * qty
                    total_outlay = net_debit + tot_spread_cost.total_cost
                    if cash >= total_outlay:
                        cash -= total_outlay
                        pos = Position(
                            pos_id=pos_id,
                            symbol=symbol,
                            segment=seg,
                            entry_time=curr_dt,
                            entry_bar_idx=i,
                            entry_price=round(p1 - p2, 2),
                            quantity=qty,
                            is_long=True,
                            legs=legs,
                            stop_loss=sig.stop_loss_pct,
                            target=sig.target_pct,
                            accumulated_costs=tot_spread_cost,
                            thesis=sig.thesis,
                        )
                        open_positions.append(pos)

                else:
                    # Equity or Futures
                    is_fut = seg == "FUTURES"
                    margin_pct = 0.15 if is_fut else 1.0
                    cost_seg = "FUTURES" if is_fut else "EQUITY_DELIVERY"
                    qty = lot_size if is_fut else max(10, int((cash * 0.1) / exec_price))
                    order_cost = self.cost_model.calculate_order_cost("BUY", exec_price, qty, segment=cost_seg)

                    req_cash = (exec_price * qty * margin_pct) + order_cost.total_cost
                    if cash >= req_cash:
                        cash -= req_cash
                        pos = Position(
                            pos_id=pos_id,
                            symbol=symbol,
                            segment=seg,
                            entry_time=curr_dt,
                            entry_bar_idx=i,
                            entry_price=exec_price,
                            quantity=qty,
                            is_long=(sig.action == "ENTER_LONG"),
                            stop_loss=sig.stop_loss_pct,
                            target=sig.target_pct,
                            current_price=exec_price,
                            accumulated_costs=order_cost,
                            thesis=sig.thesis,
                        )
                        open_positions.append(pos)

                pending_signal = None

            # -------------------------------------------------------------
            # STEP 2: Update Mark-to-Market Valuations for Open Positions
            # -------------------------------------------------------------
            for pos in open_positions:
                cur_opt_p = None
                if pos.segment.startswith("OPTION"):
                    bars_left = max(1, pos.entry_bar_idx + 7 - i)
                    cur_opt_p = BlackScholesEngine.price(spot_close, pos.strike, (bars_left/365.0), opt_type=pos.option_type)
                pos.update_valuation(spot_close, cur_opt_p)

            # -------------------------------------------------------------
            # STEP 3: Evaluate Exits on Open Positions
            # -------------------------------------------------------------
            remaining_positions: List[Position] = []
            for pos in open_positions:
                should_exit, exit_reason = self.strategy.evaluate_exit(pos, curr_bar, market_data, i)

                if should_exit:
                    # Execute Exit
                    exit_p = pos.current_price
                    seg = pos.segment
                    cost_seg = "OPTIONS" if ("OPTION" in seg or "SPREAD" in seg) else ("FUTURES" if seg == "FUTURES" else "EQUITY_DELIVERY")
                    action_exit = "SELL" if pos.is_long else "BUY"

                    cost_exit = self.cost_model.calculate_order_cost(action_exit, exit_p, pos.quantity, segment=cost_seg)
                    total_trade_costs = pos.accumulated_costs + cost_exit

                    gross_pnl = pos.unrealized_pnl
                    net_pnl = gross_pnl - total_trade_costs.total_cost

                    # Return capital outlay + gross P&L - exit order costs
                    # (Note: entry costs were already deducted from cash at execution)
                    cash += pos.capital_outlay + gross_pnl - cost_exit.total_cost

                    ret_pct = pos.unrealized_pnl_pct

                    legs_desc = ""
                    if pos.legs:
                        legs_desc = " | ".join(f"{l['action']} {l['strike']:.0f}{l['type']}" for l in pos.legs)

                    closed_trades.append(Trade(
                        trade_id=f"TRD_{len(closed_trades)+1:04d}",
                        symbol=pos.symbol,
                        segment=pos.segment,
                        action="LONG" if pos.is_long else "SHORT",
                        entry_time=pos.entry_time,
                        exit_time=curr_dt,
                        entry_price=pos.entry_price,
                        exit_price=exit_p,
                        quantity=pos.quantity,
                        holding_bars=(i - pos.entry_bar_idx),
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        return_pct=ret_pct,
                        exit_reason=exit_reason,
                        costs=total_trade_costs,
                        legs_summary=legs_desc,
                    ))
                else:
                    remaining_positions.append(pos)

            open_positions = remaining_positions

            # -------------------------------------------------------------
            # STEP 4: Record Equity Snapshot
            # -------------------------------------------------------------
            unrealized = sum(p.unrealized_pnl for p in open_positions)
            locked_capital = sum(p.capital_outlay for p in open_positions)
            current_equity = cash + locked_capital + unrealized

            equity_records.append({
                "datetime": curr_dt,
                "cash": round(cash, 2),
                "unrealized_pnl": round(unrealized, 2),
                "equity": round(current_equity, 2),
                "open_positions": len(open_positions),
                "spot_close": spot_close,
            })

            # -------------------------------------------------------------
            # STEP 5: Strategy Signal Evaluation (Strictly Bar t, Executed at t+1)
            # -------------------------------------------------------------
            if len(open_positions) < self.max_active_positions and pending_signal is None:
                sig = self.strategy.generate_signals(
                    market_data,
                    PortfolioState(
                        timestamp=curr_dt,
                        cash=cash,
                        starting_capital=self.initial_capital,
                        open_positions=open_positions,
                        equity=current_equity,
                    ),
                    as_of_idx=i,
                )
                if sig is not None:
                    pending_signal = sig
                    pending_signal_bar = i

        # Close any lingering positions at end of backtest period
        if open_positions:
            last_dt = timestamps[-1]
            last_bar = market_data.iloc[-1]
            for pos in open_positions:
                seg = pos.segment
                cost_seg = "OPTIONS" if ("OPTION" in seg or "SPREAD" in seg) else ("FUTURES" if seg == "FUTURES" else "EQUITY_DELIVERY")
                action_exit = "SELL" if pos.is_long else "BUY"
                cost_exit = self.cost_model.calculate_order_cost(action_exit, pos.current_price, pos.quantity, segment=cost_seg)
                total_trade_costs = pos.accumulated_costs + cost_exit
                gross_pnl = pos.unrealized_pnl
                net_pnl = gross_pnl - total_trade_costs.total_cost

                # Return capital outlay + gross P&L - exit order costs
                cash += pos.capital_outlay + gross_pnl - cost_exit.total_cost

                closed_trades.append(Trade(
                    trade_id=f"TRD_{len(closed_trades)+1:04d}",
                    symbol=pos.symbol,
                    segment=pos.segment,
                    action="LONG" if pos.is_long else "SHORT",
                    entry_time=pos.entry_time,
                    exit_time=last_dt,
                    entry_price=pos.entry_price,
                    exit_price=pos.current_price,
                    quantity=pos.quantity,
                    holding_bars=(len(market_data) - 1 - pos.entry_bar_idx),
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    return_pct=pos.unrealized_pnl_pct,
                    exit_reason="End of Backtest Horizon",
                    costs=total_trade_costs,
                ))

            if equity_records:
                equity_records[-1]["cash"] = round(cash, 2)
                equity_records[-1]["unrealized_pnl"] = 0.0
                equity_records[-1]["equity"] = round(cash, 2)
                equity_records[-1]["open_positions"] = 0

        df_equity = pd.DataFrame(equity_records).set_index("datetime")

        return {
            "initial_capital": self.initial_capital,
            "final_equity": df_equity["equity"].iloc[-1] if not df_equity.empty else self.initial_capital,
            "equity_curve": df_equity,
            "trades": closed_trades,
            "total_trades": len(closed_trades),
            "strategy_name": self.strategy.name,
            "strategy_version": self.strategy.version,
            "symbol": symbol,
        }
