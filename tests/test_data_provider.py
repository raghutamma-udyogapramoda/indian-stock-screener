"""
Comprehensive Unit Tests for Historical Market Data Infrastructure & Derivatives Pricing.
Covers YahooHistoricalDataProvider, LocalHistoricalDataProvider, DerivativesDataProvider,
Black-Scholes analytical option pricing, Greeks sensitivity, and DataQualityReport provenance.
"""
import unittest
import math
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.options_analytics import BlackScholesEngine
from core.backtesting.data_provider import (
    HistoricalDataProvider,
    YahooHistoricalDataProvider,
    LocalHistoricalDataProvider,
    DerivativesDataProvider,
    DataQualityReport,
)


class TestDataQualityReport(unittest.TestCase):
    def test_report_provenance_and_dict(self):
        report = DataQualityReport(
            instrument="NIFTY",
            start_date="2023-01-01",
            end_date="2023-12-31",
            total_bars=250,
            verified_bars=250,
            simulated_bars=0,
            missing_bars=0,
            duplicates_removed=2,
            data_source="YAHOO_EXCHANGE_VERIFIED",
            is_verified=True,
            warnings=["2 duplicate timestamps removed"],
        )
        d = report.to_dict()
        self.assertEqual(d["instrument"], "NIFTY")
        self.assertEqual(d["total_bars"], 250)
        self.assertTrue(d["is_verified"])
        self.assertEqual(len(d["warnings"]), 1)


class TestYahooHistoricalDataProvider(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.provider = YahooHistoricalDataProvider(cache_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_symbol_ticker_mapping(self):
        self.assertEqual(self.provider.map_symbol_to_ticker("NIFTY"), "^NSEI")
        self.assertEqual(self.provider.map_symbol_to_ticker("NIFTY 50"), "^NSEI")
        self.assertEqual(self.provider.map_symbol_to_ticker("BANKNIFTY"), "^NSEBANK")
        self.assertEqual(self.provider.map_symbol_to_ticker("SENSEX"), "^BSESN")
        self.assertEqual(self.provider.map_symbol_to_ticker("FINNIFTY"), "NIFTY_FIN_SERVICE.NS")
        self.assertEqual(self.provider.map_symbol_to_ticker("RELIANCE"), "RELIANCE.NS")
        self.assertEqual(self.provider.map_symbol_to_ticker("TCS.NS"), "TCS.NS")

    def test_contract_specifications(self):
        nifty_spec = self.provider.get_contract_specifications("NIFTY")
        self.assertEqual(nifty_spec["symbol"], "NIFTY")
        self.assertEqual(nifty_spec["strike_step"], 50)
        self.assertEqual(nifty_spec["margin_pct"], 0.15)

        banknifty_spec = self.provider.get_contract_specifications("BANKNIFTY")
        self.assertEqual(banknifty_spec["symbol"], "BANKNIFTY")
        self.assertEqual(banknifty_spec["strike_step"], 100)

        equity_spec = self.provider.get_contract_specifications("RELIANCE")
        self.assertEqual(equity_spec["symbol"], "RELIANCE")
        self.assertGreater(equity_spec["lot_size"], 0)

    def test_parquet_caching_mechanism(self):
        # Create a mock parquet file in temp cache
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame({
            "open": [18000.0] * 10,
            "high": [18100.0] * 10,
            "low": [17950.0] * 10,
            "close": [18050.0] * 10,
            "volume": [1000000] * 10,
        }, index=dates)

        cache_path = Path(self.temp_dir) / "NSEI_1d.parquet"
        mock_df.to_parquet(cache_path)

        # Retrieve through provider
        df, report = self.provider.get_historical_bars("NIFTY", "2023-01-02", "2023-01-08")
        self.assertEqual(len(df), 7)
        self.assertTrue(report.is_verified)
        self.assertEqual(report.data_source, "LOCAL_VERIFIED_PARQUET_CACHE")


class TestLocalHistoricalDataProvider(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.provider = LocalHistoricalDataProvider(data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_csv_returns_unverified_report(self):
        df, report = self.provider.get_historical_bars("MISSING_SYM", "2023-01-01", "2023-02-01")
        self.assertTrue(df.empty)
        self.assertFalse(report.is_verified)
        self.assertEqual(report.data_source, "LOCAL_CSV_MISSING")
        self.assertGreater(len(report.warnings), 0)

    def test_existing_csv_loading(self):
        csv_file = Path(self.temp_dir) / "TESTSYM_1d.csv"
        dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
        df_raw = pd.DataFrame({
            "Date": dates,
            "Open": [100.0, 102.0, 101.0],
            "High": [105.0, 104.0, 103.0],
            "Low": [98.0, 100.0, 99.0],
            "Close": [102.0, 101.0, 102.5],
            "Volume": [50000, 60000, 55000],
        })
        df_raw.to_csv(csv_file, index=False)

        df, report = self.provider.get_historical_bars("TESTSYM", "2023-01-01", "2023-01-03")
        self.assertEqual(len(df), 3)
        self.assertTrue(report.is_verified)
        self.assertIn("close", df.columns)


class TestDerivativesPricingModel(unittest.TestCase):
    def setUp(self):
        self.deriv_provider = DerivativesDataProvider(risk_free_rate=0.065, baseline_iv=0.15)

    def test_black_scholes_deep_itm_and_otm(self):
        # Deep ITM Call (Spot 24000, Strike 20000, T=30 days) -> Value close to intrinsic (4000)
        c_deep_itm = BlackScholesEngine.price(24000, 20000, 30.0 / 365.0, r=0.065, iv=0.15, opt_type="CE")
        self.assertGreater(c_deep_itm, 4000.0)

        # Deep OTM Call (Spot 24000, Strike 28000, T=30 days) -> Value close to 0
        c_deep_otm = BlackScholesEngine.price(24000, 28000, 30.0 / 365.0, r=0.065, iv=0.15, opt_type="CE")
        self.assertLess(c_deep_otm, 10.0)

        # Deep ITM Put (Spot 24000, Strike 28000, T=30 days) -> Value close to discounted intrinsic (~3850)
        p_deep_itm = BlackScholesEngine.price(24000, 28000, 30.0 / 365.0, r=0.065, iv=0.15, opt_type="PE")
        self.assertGreater(p_deep_itm, 3800.0)

    def test_expiry_boundary_convergence(self):
        # At expiry (T -> 0.0001), Call price must converge to max(0, Spot - Strike)
        spot = 24150.0
        strike = 24100.0
        c_exp = BlackScholesEngine.price(spot, strike, 0.0001, r=0.065, iv=0.15, opt_type="CE")
        self.assertAlmostEqual(c_exp, 50.0, delta=2.5)

    def test_get_option_history_synthesis(self):
        # Generate underlying bars
        dates = pd.date_range("2023-01-01", periods=15, freq="D")
        underlying_df = pd.DataFrame({
            "open": [24000.0] * 15,
            "high": [24100.0] * 15,
            "low": [23950.0] * 15,
            "close": [24050.0] * 15,
            "volume": [1000000] * 15,
        }, index=dates)

        expiry_date = "2023-01-20"
        df_opt, report = self.deriv_provider.get_option_history(
            underlying="NIFTY",
            strike=24000.0,
            option_type="CE",
            expiry=expiry_date,
            start_date="2023-01-01",
            end_date="2023-01-15",
            underlying_df=underlying_df,
        )

        self.assertFalse(df_opt.empty)
        self.assertEqual(len(df_opt), 15)
        # Provenance: explicitly not verified raw tick, tagged BS_ANALYTICAL_SIMULATION
        self.assertFalse(report.is_verified)
        self.assertEqual(report.data_source, "BS_ANALYTICAL_SIMULATION")
        self.assertIn("delta", df_opt.columns)
        self.assertIn("theta", df_opt.columns)
        self.assertIn("iv", df_opt.columns)


if __name__ == "__main__":
    unittest.main()
