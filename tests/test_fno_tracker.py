"""
Unit tests for FNOTracker and PortfolioTracker watchlist item tracking.
"""
import unittest
import sys
import shutil
import tempfile
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.fno_tracker import FNOTracker, FNO_TRACKER_DIR
from core.tracker import PortfolioTracker


class TestFNOTrackerWatchlist(unittest.TestCase):
    def setUp(self):
        self.test_user = "test_fno_unit_trader"
        FNOTracker.clear_all_positions(self.test_user)
        PortfolioTracker.clear_all_positions(self.test_user)

    def tearDown(self):
        FNOTracker.clear_all_positions(self.test_user)
        PortfolioTracker.clear_all_positions(self.test_user)

    def test_fno_tracker_add_watchlist_item_position(self):
        """Verify add_watchlist_item alias on FNOTracker functions correctly."""
        res = FNOTracker.add_watchlist_item(
            username=self.test_user,
            symbol="NIFTY",
            contract_type="FUTURES",
            strike_price=0.0,
            expiry_date="29-OCT-2026",
            lot_size=65,
            lots=1,
            buy_price=24500.0,
            stop_loss=24300.0,
            target=24900.0,
            notes="High-Conviction Futures BUY"
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "NIFTY")
        self.assertEqual(res["contract_type"], "FUTURES")
        self.assertEqual(res["expiry"], "29-OCT-2026")
        self.assertEqual(res["lot_size"], 65)
        self.assertEqual(res["buy_price"], 24500.0)

        # Ensure loaded positions contains it
        positions = FNOTracker.load_positions(self.test_user)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "NIFTY")

    def test_fno_tracker_add_position_with_full_details(self):
        """Verify add_position with full details works as expected."""
        res = FNOTracker.add_position(
            username=self.test_user,
            symbol="RELIANCE",
            contract_type="FUTURES",
            strike=0.0,
            expiry="29-OCT-2026",
            lot_size=250,
            lots=1,
            buy_price=2800.0,
            stop_loss=2750.0,
            target=2920.0,
            notes="Institutional Breakout"
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "RELIANCE")
        self.assertEqual(res["buy_price"], 2800.0)
        self.assertEqual(res["stop_loss"], 2750.0)
        self.assertEqual(res["target"], 2920.0)

    def test_portfolio_tracker_add_watchlist_item(self):
        """Verify PortfolioTracker.add_watchlist_item alias."""
        res = PortfolioTracker.add_watchlist_item(
            username=self.test_user,
            symbol="INFY",
            buy_price=1600.0,
            qty=10,
            notes="Test INFY equity"
        )
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "INFY")
        self.assertEqual(res["buy_price"], 1600.0)


if __name__ == "__main__":
    unittest.main()
