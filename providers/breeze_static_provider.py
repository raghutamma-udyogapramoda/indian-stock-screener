"""
Static / Mock Breeze API Data Provider.
Enables offline testing, CI testing, and deterministic screening for Breakout and Breakdown candidates
across NSE Equities and MCX Commodities without requiring a live, active ICICI Direct session token.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config.settings import CACHE_DIR, STATIC_BREEZE_DIR
from core.universe import UniverseManager
from providers.base import BaseDataProvider
from providers.breeze_provider import BreezeProvider


class BreezeStaticProvider(BaseDataProvider):
    """
    Offline Data Provider reading static Breeze API JSON files.
    Allows testing Breakout and Breakdown candidate detection on both NSE equities and MCX commodities.
    """

    def __init__(self, static_dir: Path = STATIC_BREEZE_DIR):
        super().__init__()
        self.is_latest = False
        self.data_source_mode = "STATIC_TEST"
        self.api_failure_reasons.append("Offline static test fixtures active. Data is generated/cached for testing and is not latest.")
        self.static_dir = Path(static_dir)
        self.static_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_sample_fixtures()

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1day",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Loads OHLCV data from static Breeze JSON fixture.
        Fallback order:
          1. data/static_breeze/<clean_symbol>.json
          2. data/cache/<clean_symbol>.parquet
          3. Generate realistic synthetic fixture for the symbol
        """
        clean_symbol = UniverseManager.to_clean_symbol(symbol)
        json_path = self.static_dir / f"{clean_symbol}.json"

        # 1. Check static Breeze JSON fixture
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = BreezeProvider.parse_breeze_response(data)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass

        # 2. Check local parquet cache
        cache_file = CACHE_DIR / f"{clean_symbol}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if df is not None and len(df) >= 30:
                    return df
            except Exception:
                pass

        # 3. If neither exists, generate and persist realistic test fixture
        df = self._generate_fixture(clean_symbol)
        return df

    def fetch_batch_ohlcv(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1day",
        max_workers: int = 5,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        for sym in symbols:
            df = self.fetch_ohlcv(sym, period, interval, use_cache=use_cache)
            if df is not None:
                results[UniverseManager.to_clean_symbol(sym)] = df
        return results

    def fetch_quote(self, symbol: str) -> Optional[dict]:
        df = self.fetch_ohlcv(symbol)
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        clean = UniverseManager.to_clean_symbol(symbol)
        exchange = "MCX" if UniverseManager.is_commodity(clean) else "NSE"

        return {
            "symbol": clean,
            "exchange": exchange,
            "last_price": float(last["close"]),
            "open": float(last["open"]),
            "high": float(last["high"]),
            "low": float(last["low"]),
            "prev_close": float(prev["close"]),
            "volume": float(last["volume"]),
            "fifty_two_week_high": float(df["high"].tail(252).max()),
            "fifty_two_week_low": float(df["low"].tail(252).min()),
        }

    def ensure_sample_fixtures(self):
        """Creates sample Breeze API JSON fixtures for breakout, breakdown, and control assets."""
        fixtures = [
            ("GOLD", "BREAKOUT", 72000.0, 76500.0, 8000),         # MCX Commodity Breakout (~₹76,500 / 10g)
            ("TRENT", "BREAKOUT", 4200.0, 7250.0, 450000),        # NSE Equity Breakout
            ("CRUDEOIL", "BREAKDOWN", 6800.0, 5850.0, 12000),      # MCX Commodity Breakdown (~₹5,850 / bbl)
            ("ASIANPAINT", "BREAKDOWN", 3100.0, 2280.0, 850000),   # NSE Equity Breakdown
            ("RELIANCE", "NEUTRAL", 1250.0, 1265.0, 3200000),      # NSE Equity Neutral
            ("SILVER", "CONSOLIDATION", 88000.0, 88200.0, 15000),  # MCX Commodity Consolidation (~₹88,200 / kg)
            ("NATURALGAS", "NEUTRAL", 260.0, 235.0, 25000),       # MCX Commodity (~₹235 / mmBtu)
            ("COPPER", "NEUTRAL", 780.0, 825.0, 18000),           # MCX Commodity (~₹825 / kg)
            ("ZINC", "NEUTRAL", 250.0, 272.0, 14000),             # MCX Commodity (~₹272 / kg)
            ("ALUMINIUM", "NEUTRAL", 220.0, 241.0, 16000),        # MCX Commodity (~₹241 / kg)
        ]

        for symbol, pattern, base_price, current_price, avg_vol in fixtures:
            json_path = self.static_dir / f"{symbol}.json"
            if not json_path.exists():
                self._save_static_breeze_json(symbol, pattern, base_price, current_price, avg_vol)

    def _generate_fixture(self, symbol: str) -> pd.DataFrame:
        """Generates and writes a default fixture if a requested symbol is missing."""
        comm_defaults = {
            "GOLD": (72000.0, 76500.0, 8000),
            "SILVER": (85000.0, 90200.0, 15000),
            "CRUDEOIL": (6800.0, 5850.0, 12000),
            "NATURALGAS": (260.0, 235.0, 25000),
            "COPPER": (780.0, 825.0, 18000),
            "ZINC": (250.0, 272.0, 14000),
            "ALUMINIUM": (220.0, 241.0, 16000),
        }
        if symbol in comm_defaults:
            base, final, vol = comm_defaults[symbol]
        else:
            is_comm = UniverseManager.is_commodity(symbol)
            base = 1000.0 if is_comm else 1500.0
            final = base * 1.05
            vol = 10000 if is_comm else 200000
        self._save_static_breeze_json(symbol, "NEUTRAL", base, final, vol)
        with open(self.static_dir / f"{symbol}.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return BreezeProvider.parse_breeze_response(data)

    def _save_static_breeze_json(
        self,
        symbol: str,
        pattern: str,
        base_price: float,
        final_price: float,
        avg_volume: int,
        num_candles: int = 260
    ):
        """
        Synthesizes 260 daily candles formatted exactly as ICICI Direct Breeze API's
        get_historical_data_v2 JSON response.
        """
        np.random.seed(abs(hash(symbol)) % 10000)
        end_date = datetime.now()
        # Generate trading dates (excluding weekends)
        dates = []
        cur = end_date
        while len(dates) < num_candles:
            if cur.weekday() < 5:  # Monday to Friday
                dates.append(cur)
            cur -= timedelta(days=1)
        dates.reverse()

        prices = []
        if pattern == "BREAKOUT":
            # 200 days base building, 40 days consolidation squeeze near resistance, final 2 days breakout surge
            t = np.linspace(0, 1, num_candles)
            trend = base_price + (final_price * 0.95 - base_price) * (t ** 1.3)
            # Add volatility
            noise = np.random.normal(0, base_price * 0.007, num_candles)
            prices = trend + noise
            # Consolidation in candles -40 to -3 (tight band near resistance)
            res_level = final_price * 0.96
            prices[-40:-3] = res_level + np.random.normal(0, base_price * 0.003, 37)
            # Candle -2 touches resistance
            prices[-2] = res_level * 1.002
            # Final candle breaks out decisively
            prices[-1] = final_price
        elif pattern == "BREAKDOWN":
            # Downtrend breaking below key horizontal floor
            t = np.linspace(0, 1, num_candles)
            trend = base_price - (base_price - final_price * 1.06) * (t ** 1.1)
            noise = np.random.normal(0, base_price * 0.008, num_candles)
            prices = trend + noise
            # Support floor in candles -30 to -3
            supp_level = final_price * 1.05
            prices[-30:-3] = supp_level + np.abs(np.random.normal(0, base_price * 0.003, 27))
            # Final candle breaks support down with high volume
            prices[-2] = supp_level * 0.995
            prices[-1] = final_price
        else:
            # Neutral / Oscillating
            t = np.linspace(0, 4 * np.pi, num_candles)
            trend = base_price + (final_price - base_price) * 0.5 + np.sin(t) * (base_price * 0.03)
            noise = np.random.normal(0, base_price * 0.005, num_candles)
            prices = trend + noise

        success_records = []
        for i, dt in enumerate(dates):
            close = float(prices[i])
            daily_range = close * 0.015
            if i == num_candles - 1 and pattern == "BREAKOUT":
                # High CLV close near high
                low = close - daily_range * 0.9
                high = close + daily_range * 0.1
                open_p = low + daily_range * 0.2
                vol = int(avg_volume * 2.2)
            elif i == num_candles - 1 and pattern == "BREAKDOWN":
                # Low CLV close near low
                high = close + daily_range * 0.9
                low = close - daily_range * 0.1
                open_p = high - daily_range * 0.2
                vol = int(avg_volume * 2.0)
            else:
                low = close - abs(np.random.normal(daily_range * 0.5, daily_range * 0.2))
                high = close + abs(np.random.normal(daily_range * 0.5, daily_range * 0.2))
                open_p = low + (high - low) * np.random.uniform(0.2, 0.8)
                vol = int(avg_volume * np.random.uniform(0.7, 1.3))

            # Ensure valid OHLC
            high = max(high, open_p, close)
            low = min(low, open_p, close)

            success_records.append({
                "datetime": dt.strftime("%Y-%m-%d 09:15:00"),
                "open": round(float(open_p), 2),
                "high": round(float(high), 2),
                "low": round(float(low), 2),
                "close": round(float(close), 2),
                "volume": int(max(vol, 100))
            })

        payload = {
            "Success": success_records,
            "Status": 200,
            "Error": None
        }

        output_file = self.static_dir / f"{symbol}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
