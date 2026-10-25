"""
Comprehensive Unit Tests for Indian Statutory Costs & Taxation Engine.
Covers Equity Delivery, Equity Intraday, Futures, Options, Multi-Leg Spreads,
Post-Budget 2024 STT rates, GST, Stamp Duty, Slippage, and boundary edge cases.
"""
import unittest
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.backtesting.costs import IndianCostModel, CostBreakdown


class TestIndianCostModelDetailed(unittest.TestCase):
    def setUp(self):
        self.model = IndianCostModel(
            brokerage_per_order=20.0,
            brokerage_pct_cap=0.0003,
            slippage_pct=0.0005,
            enable_statutory=True,
        )

    def test_zero_quantity_or_price(self):
        # Trade with 0 qty or 0 price should have 0 cost
        cost_zero_qty = self.model.calculate_order_cost("BUY", 100.0, 0, segment="EQUITY_DELIVERY")
        self.assertEqual(cost_zero_qty.total_cost, 0.0)
        self.assertEqual(cost_zero_qty.brokerage, 0.0)

        cost_zero_price = self.model.calculate_order_cost("BUY", 0.0, 50, segment="OPTIONS")
        self.assertEqual(cost_zero_price.total_cost, 0.0)

    def test_equity_delivery_costs(self):
        # 100 shares of RELIANCE bought at ₹2,500 = ₹250,000 turnover
        cost_buy = self.model.calculate_order_cost(
            action="BUY", price=2500.0, quantity=100, segment="EQUITY_DELIVERY"
        )
        # Brokerage is 0 for delivery in discount model
        self.assertEqual(cost_buy.brokerage, 0.0)
        # STT on Delivery Buy is 0.1% of turnover = 250,000 * 0.001 = ₹250.0
        self.assertAlmostEqual(cost_buy.stt, 250.0, places=2)
        # Stamp Duty on Delivery Buy is 0.015% of turnover = 250,000 * 0.00015 = ₹37.5
        self.assertAlmostEqual(cost_buy.stamp_duty, 37.5, places=2)
        # Slippage: 250,000 * 0.0005 = ₹125.0
        self.assertAlmostEqual(cost_buy.slippage, 125.0, places=2)

        # 100 shares of RELIANCE sold at ₹2,600 = ₹260,000 turnover
        cost_sell = self.model.calculate_order_cost(
            action="SELL", price=2600.0, quantity=100, segment="EQUITY_DELIVERY"
        )
        # STT on Delivery Sell is 0.1% = 260,000 * 0.001 = ₹260.0
        self.assertAlmostEqual(cost_sell.stt, 260.0, places=2)
        # Stamp duty on Sell is 0.0
        self.assertEqual(cost_sell.stamp_duty, 0.0)

    def test_equity_intraday_costs(self):
        # 200 shares of INFY bought at ₹1,500 = ₹300,000 turnover
        cost_buy = self.model.calculate_order_cost(
            action="BUY", price=1500.0, quantity=200, segment="EQUITY_INTRADAY"
        )
        # Brokerage is min(20, 300,000 * 0.0003 = 90) = ₹20.0
        self.assertEqual(cost_buy.brokerage, 20.0)
        # STT on Intraday Buy is 0.0
        self.assertEqual(cost_buy.stt, 0.0)
        # Stamp duty on Intraday Buy is 0.003% = 300,000 * 0.00003 = ₹9.0
        self.assertAlmostEqual(cost_buy.stamp_duty, 9.0, places=2)

        # 200 shares sold at ₹1,520 = ₹304,000 turnover
        cost_sell = self.model.calculate_order_cost(
            action="SELL", price=1520.0, quantity=200, segment="EQUITY_INTRADAY"
        )
        # STT on Intraday Sell is 0.025% = 304,000 * 0.00025 = ₹76.0
        self.assertAlmostEqual(cost_sell.stt, 76.0, places=2)

    def test_futures_costs(self):
        # 1 lot NIFTY futures (lot size 25) bought at ₹24,000 = ₹600,000 turnover
        cost_buy = self.model.calculate_order_cost(
            action="BUY", price=24000.0, quantity=25, segment="FUTURES"
        )
        # STT on Futures Buy is 0.0
        self.assertEqual(cost_buy.stt, 0.0)
        # Stamp duty on Futures Buy is 0.002% = 600,000 * 0.00002 = ₹12.0
        self.assertAlmostEqual(cost_buy.stamp_duty, 12.0, places=2)

        # Sold at ₹24,500 = ₹612,500 turnover
        cost_sell = self.model.calculate_order_cost(
            action="SELL", price=24500.0, quantity=25, segment="FUTURES"
        )
        # Post-Budget 2024 STT on Futures Sell: 0.02% = 612,500 * 0.0002 = ₹122.50
        self.assertAlmostEqual(cost_sell.stt, 122.50, places=2)
        self.assertEqual(cost_sell.stamp_duty, 0.0)

    def test_options_post_budget_2024_stt(self):
        # 2 lots NIFTY Options (50 qty) bought at ₹120 = ₹6,000 premium turnover
        cost_buy = self.model.calculate_order_cost(
            action="BUY", price=120.0, quantity=50, segment="OPTIONS"
        )
        self.assertEqual(cost_buy.brokerage, 20.0)
        self.assertEqual(cost_buy.stt, 0.0)
        # Stamp duty on Options Buy is 0.003% = 6,000 * 0.00003 = ₹0.18
        self.assertAlmostEqual(cost_buy.stamp_duty, 0.18, places=2)

        # 2 lots sold at ₹180 = ₹9,000 premium turnover
        cost_sell = self.model.calculate_order_cost(
            action="SELL", price=180.0, quantity=50, segment="OPTIONS"
        )
        # Post-Budget 2024 STT on Options Sell is 0.1% of premium turnover:
        # 9,000 * 0.001 = ₹9.00
        self.assertAlmostEqual(cost_sell.stt, 9.00, places=2)
        self.assertEqual(cost_sell.brokerage, 20.0)
        self.assertEqual(cost_sell.stamp_duty, 0.0)

    def test_multi_leg_spread_addition(self):
        # Bull Call Spread: Buy Leg + Sell Leg
        leg_buy = self.model.calculate_order_cost("BUY", 150.0, 25, segment="OPTIONS")
        leg_sell = self.model.calculate_order_cost("SELL", 80.0, 25, segment="OPTIONS")

        combined = leg_buy + leg_sell
        self.assertEqual(combined.brokerage, 40.0)  # 2 orders * 20
        self.assertEqual(combined.stt, leg_sell.stt)
        self.assertAlmostEqual(combined.total_cost, leg_buy.total_cost + leg_sell.total_cost, places=2)

    def test_statutory_disabled_mode(self):
        no_stat_model = IndianCostModel(brokerage_per_order=15.0, enable_statutory=False)
        cost = no_stat_model.calculate_order_cost("SELL", 200.0, 50, segment="OPTIONS")
        self.assertEqual(cost.brokerage, 15.0)
        self.assertEqual(cost.stt, 0.0)
        self.assertEqual(cost.gst, 0.0)
        self.assertEqual(cost.stamp_duty, 0.0)
        self.assertEqual(cost.exchange_charges, 0.0)
        self.assertAlmostEqual(cost.total_cost, 15.0 + cost.slippage, places=2)

    def test_slippage_override(self):
        cost_default = self.model.calculate_order_cost("BUY", 100.0, 100, segment="OPTIONS")
        cost_high_slip = self.model.calculate_order_cost(
            "BUY", 100.0, 100, segment="OPTIONS", slippage_override_pct=0.005
        )
        self.assertGreater(cost_high_slip.slippage, cost_default.slippage)
        self.assertAlmostEqual(cost_high_slip.slippage, 10000.0 * 0.005, places=2)


if __name__ == "__main__":
    unittest.main()
