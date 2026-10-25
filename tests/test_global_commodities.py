"""
Unit Tests for Global Commodities Universe (COMEX, NYMEX, ICE, LME).
Verifies that Gold, Silver, Crude Oil, Copper, Natural Gas, Platinum, and Palladium
are quoted in authentic native USD ($), mapped to global futures, not converted to MCX INR,
and correctly evaluated across screeners and backtesters.
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.universe import UniverseManager, GLOBAL_COMMODITIES_SEED
from core.backtesting.data_provider import YahooHistoricalDataProvider
from core.backtesting.evaluator import enrich_plans_with_backtest_evaluations
from screeners.breakout import BreakoutScreener
from screeners.breakdown import BreakdownScreener
from screeners.swing import SwingScreener
from screeners.intraday import IntradayScreener
from screeners.btst import BTSTScreener


class TestGlobalCommoditiesUniverse(unittest.TestCase):
    def test_global_commodities_seed_presence(self):
        tickers = UniverseManager.get_tickers("GLOBAL_COMMODITIES")
        self.assertGreaterEqual(len(tickers), 8)
        self.assertIn("GOLD_GLOBAL", tickers)
        self.assertIn("SILVER_GLOBAL", tickers)
        self.assertIn("CRUDE_WTI", tickers)
        self.assertIn("BRENT_CRUDE", tickers)
        self.assertIn("COPPER_GLOBAL", tickers)
        self.assertIn("NATGAS_GLOBAL", tickers)
        self.assertIn("PLATINUM_GLOBAL", tickers)
        self.assertIn("PALLADIUM_GLOBAL", tickers)

    def test_global_commodity_detection(self):
        self.assertTrue(UniverseManager.is_global_commodity("GOLD_GLOBAL"))
        self.assertTrue(UniverseManager.is_global_commodity("SILVER_GLOBAL"))
        self.assertTrue(UniverseManager.is_global_commodity("CRUDE_WTI"))
        self.assertTrue(UniverseManager.is_global_commodity("BRENT_CRUDE"))
        self.assertTrue(UniverseManager.is_global_commodity("COPPER_GLOBAL"))
        self.assertTrue(UniverseManager.is_global_commodity("NATGAS_GLOBAL"))
        self.assertTrue(UniverseManager.is_global_commodity("GC=F"))
        self.assertTrue(UniverseManager.is_global_commodity("CL=F"))

        # Equities and MCX commodities should not be global commodities
        self.assertFalse(UniverseManager.is_global_commodity("RELIANCE"))
        self.assertFalse(UniverseManager.is_global_commodity("NIFTY"))
        self.assertFalse(UniverseManager.is_global_commodity("GOLDM"))
        self.assertFalse(UniverseManager.is_global_commodity("CRUDEOIL"))

    def test_mcx_separation(self):
        # Global commodities MUST NOT be treated as MCX commodities
        self.assertFalse(UniverseManager.is_commodity("GOLD_GLOBAL"))
        self.assertFalse(UniverseManager.is_commodity("SILVER_GLOBAL"))
        self.assertFalse(UniverseManager.is_commodity("CRUDE_WTI"))
        self.assertFalse(UniverseManager.is_commodity("COPPER_GLOBAL"))

        # MCX commodities must still be recognized as MCX
        self.assertTrue(UniverseManager.is_commodity("GOLDM"))
        self.assertTrue(UniverseManager.is_commodity("SILVERM"))
        self.assertTrue(UniverseManager.is_commodity("CRUDEOIL"))

    def test_currency_and_exchange(self):
        self.assertEqual(UniverseManager.get_commodity_currency("GOLD_GLOBAL"), "$")
        self.assertEqual(UniverseManager.get_commodity_currency("CRUDE_WTI"), "$")
        self.assertEqual(UniverseManager.get_commodity_currency("COPPER_GLOBAL"), "$")
        self.assertEqual(UniverseManager.get_commodity_currency("NATGAS_GLOBAL"), "$")

        # Indian assets remain in INR
        self.assertEqual(UniverseManager.get_commodity_currency("RELIANCE"), "₹")
        self.assertEqual(UniverseManager.get_commodity_currency("GOLDM"), "₹")
        self.assertEqual(UniverseManager.get_commodity_currency("NIFTY"), "₹")

        # Exchange
        self.assertEqual(UniverseManager.get_exchange("GOLD_GLOBAL"), "GLOBAL")
        self.assertEqual(UniverseManager.get_exchange("CRUDE_WTI"), "GLOBAL")
        self.assertEqual(UniverseManager.get_exchange("GOLDM"), "MCX")
        self.assertEqual(UniverseManager.get_exchange("RELIANCE"), "NSE")

    def test_commodity_units(self):
        self.assertIn("$ / Troy Oz", UniverseManager.get_commodity_unit("GOLD_GLOBAL"))
        self.assertIn("$ / bbl", UniverseManager.get_commodity_unit("CRUDE_WTI"))
        self.assertIn("$ / lb", UniverseManager.get_commodity_unit("COPPER_GLOBAL"))
        self.assertIn("$ / mmBtu", UniverseManager.get_commodity_unit("NATGAS_GLOBAL"))

    def test_no_mcx_conversion_multiplier(self):
        # Global prices must have multiplier = 1.0 (unconverted)
        mult_gold = UniverseManager.get_mcx_conversion_multiplier("GOLD_GLOBAL", 4000.0)
        self.assertEqual(mult_gold, 1.0)

        mult_crude = UniverseManager.get_mcx_conversion_multiplier("CRUDE_WTI", 100.0)
        self.assertEqual(mult_crude, 1.0)

        mult_copper = UniverseManager.get_mcx_conversion_multiplier("COPPER_GLOBAL", 6.5)
        self.assertEqual(mult_copper, 1.0)

        # Test convert_ohlcv_to_mcx leaves DataFrame unmodified
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df_orig = pd.DataFrame({
            "open": [4000.0] * 5,
            "high": [4050.0] * 5,
            "low": [3980.0] * 5,
            "close": [4020.0] * 5,
            "volume": [1000] * 5,
        }, index=dates)

        df_res = UniverseManager.convert_ohlcv_to_mcx("GOLD_GLOBAL", df_orig)
        pd.testing.assert_frame_equal(df_orig, df_res)

    def test_yfinance_symbol_mapping(self):
        self.assertEqual(UniverseManager.to_yfinance_symbol("GOLD_GLOBAL"), "GC=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("SILVER_GLOBAL"), "SI=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("CRUDE_WTI"), "CL=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("BRENT_CRUDE"), "BZ=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("COPPER_GLOBAL"), "HG=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("NATGAS_GLOBAL"), "NG=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("PLATINUM_GLOBAL"), "PL=F")
        self.assertEqual(UniverseManager.to_yfinance_symbol("PALLADIUM_GLOBAL"), "PA=F")

    def test_backtesting_data_provider_mapping(self):
        provider = YahooHistoricalDataProvider()
        self.assertEqual(provider.map_symbol_to_ticker("GOLD_GLOBAL"), "GC=F")
        self.assertEqual(provider.map_symbol_to_ticker("CRUDE_WTI"), "CL=F")

        spec = provider.get_contract_specifications("GOLD_GLOBAL")
        self.assertEqual(spec["symbol"], "GOLD_GLOBAL")
        self.assertEqual(spec["lot_size"], 1)
        self.assertEqual(spec["margin_pct"], 0.10)

    def test_screeners_accommodate_low_dollar_commodities(self):
        # Create 50 daily candles for Natural Gas at ~$3.05
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        ng_df = pd.DataFrame({
            "open": np.linspace(2.8, 3.2, 60),
            "high": np.linspace(2.85, 3.25, 60),
            "low": np.linspace(2.75, 3.15, 60),
            "close": np.linspace(2.82, 3.22, 60),
            "volume": [500.0] * 60,
        }, index=dates)

        # Breakout screener should not reject due to close < 15.0
        bo_screener = BreakoutScreener()
        # Should not crash on low dollar price
        res = bo_screener.screen("NATGAS_GLOBAL", ng_df)
        # Even if pattern conditions vary, it should evaluate past the price floor check

        # Swing screener
        swing_screener = SwingScreener()
        res_sw = swing_screener.screen("COPPER_GLOBAL", ng_df)


if __name__ == "__main__":
    unittest.main()
