"""
ICICI Direct Breeze API Provider.
Enables real-time streaming, historical candles, and F&O / Option chain data directly
from your ICICI Direct trading account for both NSE Equities and MCX Commodities.
"""

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from config.settings import BREEZE_API_KEY, BREEZE_SECRET_KEY, BREEZE_SESSION_TOKEN
from core.cache import LocalDataCache
from core.universe import UniverseManager
from providers.base import BaseDataProvider


class BreezeProvider(BaseDataProvider):
    """Integrates ICICI Direct Breeze API for live quotes, historical data, and F&O across NSE and MCX."""

    def __init__(
        self,
        api_key: str = BREEZE_API_KEY,
        secret_key: str = BREEZE_SECRET_KEY,
        session_token: str = BREEZE_SESSION_TOKEN,
        cache_ttl_hours: float = 4.0
    ):
        super().__init__()
        self.api_key = api_key if api_key else BREEZE_API_KEY
        self.secret_key = secret_key if secret_key else BREEZE_SECRET_KEY
        self.session_token = session_token if session_token else BREEZE_SESSION_TOKEN
        self.cache = LocalDataCache(ttl_hours=cache_ttl_hours)
        self.client = None
        self._is_connected = False
        self._initialize()

    def _initialize(self):
        """Attempts to initialize BreezeConnect client if credentials exist."""
        if not (self.api_key and self.secret_key and self.session_token):
            self.api_call_failed = True
            self.is_latest = False
            self.data_source_mode = "STATIC_TEST"
            self.api_failure_reasons.append("ICICI Breeze credentials missing or session token not supplied.")
            return

        try:
            from breeze_connect import BreezeConnect
            self.client = BreezeConnect(api_key=self.api_key)
            self.client.generate_session(
                api_secret=self.secret_key,
                session_token=self.session_token
            )
            self._is_connected = True
            self.data_source_mode = "LIVE"
            self.is_latest = True
        except Exception as e:
            self._is_connected = False
            self.api_call_failed = True
            self.is_latest = False
            self.data_source_mode = "STATIC_TEST"
            self.api_failure_reasons.append(f"Failed to generate Breeze session: {type(e).__name__} ({str(e)[:100]})")

    @property
    def is_authenticated(self) -> bool:
        return self._is_connected and self.client is not None

    def _get_iso_date_range(self, period: str = "1y") -> tuple[str, str]:
        """Calculates ISO-8601 UTC date range required by Breeze API."""
        now = datetime.utcnow()
        if period == "1y":
            start = now - timedelta(days=365)
        elif period == "6mo":
            start = now - timedelta(days=180)
        elif period == "3mo":
            start = now - timedelta(days=90)
        elif period == "1mo":
            start = now - timedelta(days=30)
        elif period == "2y":
            start = now - timedelta(days=730)
        else:
            start = now - timedelta(days=365)

        from_date = start.strftime("%Y-%m-%dT07:00:00.000Z")
        to_date = now.strftime("%Y-%m-%dT07:00:00.000Z")
        return from_date, to_date

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1day",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data using Breeze API.
        Automatically detects whether the asset trades on NSE (cash equity) or MCX (commodity).
        """
        clean_symbol = UniverseManager.to_clean_symbol(symbol)
        cache_key = f"breeze_{clean_symbol}"

        # 1. Try local cache
        if use_cache:
            cached_df = self.cache.get(cache_key)
            if cached_df is not None and len(cached_df) >= 30:
                return cached_df

        if not self.is_authenticated:
            return None

        # 2. Determine exchange and product type
        is_comm = UniverseManager.is_commodity(clean_symbol)
        exchange_code = "MCX" if is_comm else "NSE"
        product_type = "commodity" if is_comm else "cash"

        from_date, to_date = self._get_iso_date_range(period)

        try:
            data = self.client.get_historical_data_v2(
                interval=interval,
                from_date=from_date,
                to_date=to_date,
                stock_code=clean_symbol,
                exchange_code=exchange_code,
                product_type=product_type
            )

            df = self.parse_breeze_response(data)
            if df is not None and not df.empty:
                if use_cache:
                    self.cache.save(cache_key, df)
                return df

            return None
        except Exception:
            return None

    @staticmethod
    def parse_breeze_response(data: dict) -> Optional[pd.DataFrame]:
        """Parses Breeze API get_historical_data_v2 JSON into standardized OHLCV DataFrame."""
        if not data or "Success" not in data or not data["Success"]:
            return None

        try:
            records = data["Success"]
            df = pd.DataFrame(records)
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "datetime": "date"
            })

            # Ensure numeric conversion
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            required = ['open', 'high', 'low', 'close', 'volume']
            return df[required].dropna()
        except Exception:
            return None

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
        """Fetches live quotes for cash equity (NSE) or commodity (MCX)."""
        if not self.is_authenticated:
            return None

        clean_symbol = UniverseManager.to_clean_symbol(symbol)
        is_comm = UniverseManager.is_commodity(clean_symbol)
        exchange_code = "MCX" if is_comm else "NSE"
        product_type = "commodity" if is_comm else "cash"

        try:
            quote = self.client.get_quotes(
                stock_code=clean_symbol,
                exchange_code=exchange_code,
                expiry_date="",
                product_type=product_type,
                right="",
                strike_price=""
            )
            if quote and "Success" in quote and quote["Success"]:
                item = quote["Success"][0]
                return {
                    "symbol": clean_symbol,
                    "exchange": exchange_code,
                    "last_price": float(item.get("ltp", 0.0)),
                    "open": float(item.get("open", 0.0)),
                    "high": float(item.get("high", 0.0)),
                    "low": float(item.get("low", 0.0)),
                    "volume": float(item.get("total_quantity_traded", 0.0)),
                }
            return None
        except Exception:
            return None

    def fetch_option_chain(self, stock_code: str, expiry_date: str) -> Optional[pd.DataFrame]:
        """Fetches Option Chain quotes for CE/PE strikes to calculate OI buildup."""
        if not self.is_authenticated:
            return None
        clean = UniverseManager.to_clean_symbol(stock_code)
        try:
            data = self.client.get_option_chain_quotes(
                stock_code=clean,
                exchange_code="NFO",
                expiry_date=expiry_date,
                product_type="options"
            )
            if data and "Success" in data:
                return pd.DataFrame(data["Success"])
            return None
        except Exception:
            return None

    def export_static_snapshot(self, symbol: str, output_dir: Path) -> bool:
        """Downloads live Breeze data and exports exact raw JSON payload for offline testing."""
        if not self.is_authenticated:
            return False

        clean = UniverseManager.to_clean_symbol(symbol)
        is_comm = UniverseManager.is_commodity(clean)
        exchange_code = "MCX" if is_comm else "NSE"
        product_type = "commodity" if is_comm else "cash"
        from_date, to_date = self._get_iso_date_range("1y")

        try:
            data = self.client.get_historical_data_v2(
                interval="1day",
                from_date=from_date,
                to_date=to_date,
                stock_code=clean,
                exchange_code=exchange_code,
                product_type=product_type
            )
            if data and "Success" in data:
                output_dir.mkdir(parents=True, exist_ok=True)
                target_file = output_dir / f"{clean}.json"
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                return True
            return False
        except Exception:
            return False
