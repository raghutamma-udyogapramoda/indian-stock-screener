"""
Indian Statutory Taxes, Exchange Charges and Realistic Execution Cost Engine.
Computes precise Brokerage, STT, Exchange Turnover Fees, GST, Stamp Duty, SEBI Charges,
and configurable Market Slippage for Equities, Futures, and Options.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class CostBreakdown:
    """Detailed ledger of all statutory and brokerage costs incurred on an order."""
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charges: float = 0.0
    gst: float = 0.0
    sebi_charges: float = 0.0
    stamp_duty: float = 0.0
    slippage: float = 0.0
    total_statutory: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "gst": round(self.gst, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "slippage": round(self.slippage, 2),
            "total_statutory": round(self.total_statutory, 2),
            "total_cost": round(self.total_cost, 2),
        }

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange_charges=self.exchange_charges + other.exchange_charges,
            gst=self.gst + other.gst,
            sebi_charges=self.sebi_charges + other.sebi_charges,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            slippage=self.slippage + other.slippage,
            total_statutory=self.total_statutory + other.total_statutory,
            total_cost=self.total_cost + other.total_cost,
        )


class IndianCostModel:
    """
    Standard Indian Securities & Derivatives Cost Model.
    Supports Equity Delivery, Equity Intraday, Futures, and Options.
    Complies with SEBI / NSE statutory rates and tax slabs.
    """

    def __init__(
        self,
        brokerage_per_order: float = 20.0,
        brokerage_pct_cap: float = 0.0003,  # 0.03% capped at ₹20
        slippage_pct: float = 0.0005,       # 0.05% baseline slippage
        enable_statutory: bool = True,
    ):
        self.brokerage_per_order = brokerage_per_order
        self.brokerage_pct_cap = brokerage_pct_cap
        self.slippage_pct = slippage_pct
        self.enable_statutory = enable_statutory

    def calculate_order_cost(
        self,
        action: str,          # "BUY" or "SELL"
        price: float,
        quantity: int,
        segment: str = "OPTIONS",  # "OPTIONS", "FUTURES", "EQUITY_DELIVERY", "EQUITY_INTRADAY"
        slippage_override_pct: Optional[float] = None,
        trade_date: Optional[datetime] = None,
    ) -> CostBreakdown:
        """
        Calculates all charges for a single executed order leg.
        Turnover = price * quantity.
        """
        if price <= 0 or quantity <= 0:
            return CostBreakdown()

        turnover = price * quantity
        action_up = action.upper()
        segment_up = segment.upper()
        slip_pct = slippage_override_pct if slippage_override_pct is not None else self.slippage_pct

        # 1. Slippage cost (Execution drag vs theoretical price)
        slippage = turnover * slip_pct

        # 2. Brokerage (e.g., Discount broker: min(0.03% turnover, ₹20))
        brokerage = min(self.brokerage_per_order, turnover * self.brokerage_pct_cap)
        if segment_up == "EQUITY_DELIVERY":
            brokerage = 0.0  # Many top Indian brokers have zero delivery brokerage

        if not self.enable_statutory:
            total_cost = brokerage + slippage
            return CostBreakdown(
                brokerage=brokerage,
                slippage=slippage,
                total_cost=total_cost,
            )

        stt = 0.0
        exchange_charges = 0.0
        stamp_duty = 0.0
        sebi_charges = turnover * 0.000001  # ₹10 per crore

        # Statutory rates based on Indian regulatory schedule:
        if segment_up == "OPTIONS":
            # Options: STT is charged on SELL option premium (0.10% post-2024 budget)
            if action_up == "SELL":
                stt = turnover * 0.0010
            # Exchange turnover charge (NSE: ~0.03503% on premium turnover)
            exchange_charges = turnover * 0.0003503
            # Stamp duty: 0.003% on BUY option premium
            if action_up == "BUY":
                stamp_duty = turnover * 0.00003

        elif segment_up == "FUTURES":
            # Futures: STT is 0.02% on SELL turnover (revised post-budget 2024)
            if action_up == "SELL":
                stt = turnover * 0.0002
            # Exchange charge: ~0.00173% on futures turnover
            exchange_charges = turnover * 0.0000173
            # Stamp duty: 0.002% on BUY turnover
            if action_up == "BUY":
                stamp_duty = turnover * 0.00002

        elif segment_up == "EQUITY_INTRADAY":
            # Intraday Equity: STT 0.025% on SELL only
            if action_up == "SELL":
                stt = turnover * 0.00025
            exchange_charges = turnover * 0.0000297
            if action_up == "BUY":
                stamp_duty = turnover * 0.00003

        else:  # EQUITY_DELIVERY
            # Delivery: STT 0.1% on both BUY and SELL
            stt = turnover * 0.001
            exchange_charges = turnover * 0.0000297
            if action_up == "BUY":
                stamp_duty = turnover * 0.00015

        # GST: 18% on (Brokerage + Exchange Charges + SEBI Charges)
        gst = (brokerage + exchange_charges + sebi_charges) * 0.18

        total_statutory = stt + exchange_charges + gst + sebi_charges + stamp_duty
        total_cost = brokerage + total_statutory + slippage

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange_charges=exchange_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage=slippage,
            total_statutory=total_statutory,
            total_cost=total_cost,
        )

    def calculate_roundtrip_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        segment: str = "OPTIONS",
        is_long: bool = True,
    ) -> CostBreakdown:
        """Calculates total roundtrip costs across both entry and exit legs."""
        entry_action = "BUY" if is_long else "SELL"
        exit_action = "SELL" if is_long else "BUY"

        c_entry = self.calculate_order_cost(entry_action, entry_price, quantity, segment)
        c_exit = self.calculate_order_cost(exit_action, exit_price, quantity, segment)

        return c_entry + c_exit
