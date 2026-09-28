"""
Yahoo Finance Data Provider.
Provides zero-setup, free access to Indian market data (NSE & BSE) using yfinance.
Ideal for weekend scans, 52-week high breakout analysis, and daily OHLCV backfilling.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

from core.cache import LocalDataCache
from core.universe import UniverseManager
from providers.base import BaseDataProvider


class YahooFinanceProvider(BaseDataProvider):
    """Fetches NSE/BSE stock data using yfinance with built-in Parquet caching and concurrency."""

    def __init__(self, cache_ttl_hours: float = 4.0):
        super().__init__()
        self.cache = LocalDataCache(ttl_hours=cache_ttl_hours)

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetches OHLCV candlestick data for a single Indian equity symbol.
        
        Data Ingestion Pipeline:
          1. Clean symbol: Strips exchange suffixes (e.g. 'RELIANCE.NS' -> 'RELIANCE').
          2. Cache Check: Checks local Parquet cache (data/cache/<SYMBOL>.parquet).
             If file exists and is younger than TTL (e.g. 4 hours), returns immediately.
          3. Network Fetch: If cache expired/missing or bypassed, fetches daily OHLCV from Yahoo Finance.
          4. Standardization: Converts columns to lowercase ['open', 'high', 'low', 'close', 'volume'].
          5. Failure Handling: If Yahoo Finance API fails, captures error reason, flags api_call_failed,
             and gracefully falls back to any existing cached parquet file on disk.
          6. Cache Write: Stores fresh DataFrame to disk as columnar Parquet file.
        """
        clean_symbol = UniverseManager.to_clean_symbol(symbol)
        
        # 1. Try local cache (within TTL)
        if use_cache:
            cached_df = self.cache.get(clean_symbol)
            if cached_df is not None and len(cached_df) >= 30:
                self.fallback_symbols.append(clean_symbol)
                return cached_df

        # 2. Fetch from external Yahoo Finance API
        yf_symbol = UniverseManager.to_yfinance_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df is None or df.empty or len(df) < 10:
                raise ValueError(f"Empty or insufficient data returned from yfinance for {yf_symbol}")

            # Standardize columns to lowercase
            df.columns = [str(c).lower() for c in df.columns]
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing required OHLCV columns from yfinance for {yf_symbol}")

            # Keep only required columns
            df = df[required].copy()
            df = df.dropna()

            # If symbol is an MCX commodity, convert international futures prices to Indian MCX terms (INR)
            if UniverseManager.is_commodity(clean_symbol):
                df = self._convert_to_mcx_inr(clean_symbol, df)

            # Cache the result
            if use_cache:
                self.cache.save(clean_symbol, df)

            self.live_symbols.append(clean_symbol)
            return df
        except Exception as e:
            # External API call failed!
            self.api_call_failed = True
            self.failed_symbols.append(clean_symbol)
            err_msg = f"{clean_symbol} ({yf_symbol}): {type(e).__name__} - {str(e)[:100]}"
            if len(self.api_failure_reasons) < 10 and err_msg not in self.api_failure_reasons:
                self.api_failure_reasons.append(err_msg)

            # FALLBACK: Try loading any existing cached parquet file from disk regardless of TTL
            any_cache = self.cache.get_any(clean_symbol)
            if any_cache is not None:
                fallback_df, age_h = any_cache
                if len(fallback_df) >= 30:
                    self.fallback_symbols.append(clean_symbol)
                    self.is_latest = False
                    self.data_source_mode = "CACHED_FALLBACK"
                    return fallback_df

            self.is_latest = False
            self.data_source_mode = "API_ERROR"
            return None

    def fetch_batch_ohlcv(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 15,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical data concurrently using a thread pool.
        Scans 500 stocks in ~15-20 seconds with local cache hits.
        """
        results: Dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_ohlcv, sym, period, interval, use_cache): sym
                for sym in symbols
            }

            for future in as_completed(future_to_symbol):
                sym = future_to_symbol[future]
                clean = UniverseManager.to_clean_symbol(sym)
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[clean] = df
                except Exception:
                    pass

        return results

    def fetch_quote(self, symbol: str) -> Optional[dict]:
        clean = UniverseManager.to_clean_symbol(symbol)
        yf_symbol = UniverseManager.to_yfinance_symbol(symbol)
        try:
            t = yf.Ticker(yf_symbol)
            fast_info = t.fast_info
            last_price = getattr(fast_info, "last_price", None) or getattr(fast_info, "regular_market_price", None)
            prev_close = getattr(fast_info, "previous_close", None)
            day_high = getattr(fast_info, "day_high", None)
            day_low = getattr(fast_info, "day_low", None)
            fifty_two_high = getattr(fast_info, "year_high", None)
            fifty_two_low = getattr(fast_info, "year_low", None)

            if UniverseManager.is_commodity(clean) and last_price:
                mult = UniverseManager.get_mcx_conversion_multiplier(clean, float(last_price))
                if abs(mult - 1.0) > 1e-4:
                    last_price = round(float(last_price) * mult, 2)
                    if prev_close:
                        prev_close = round(float(prev_close) * mult, 2)
                    if day_high:
                        day_high = round(float(day_high) * mult, 2)
                    if day_low:
                        day_low = round(float(day_low) * mult, 2)
                    if fifty_two_high:
                        fifty_two_high = round(float(fifty_two_high) * mult, 2)
                    if fifty_two_low:
                        fifty_two_low = round(float(fifty_two_low) * mult, 2)

            return {
                "symbol": clean,
                "last_price": last_price,
                "prev_close": prev_close,
                "day_high": day_high,
                "day_low": day_low,
                "fifty_two_week_high": fifty_two_high,
                "fifty_two_week_low": fifty_two_low,
            }
        except Exception:
            return None

    @staticmethod
    def _convert_to_mcx_inr(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts international US Dollar commodity futures (COMEX/NYMEX/LME)
        into official Multi Commodity Exchange of India (MCX) contract units and INR (₹).
        Delegates to UniverseManager.convert_ohlcv_to_mcx to guarantee system-wide consistency.
        """
        return UniverseManager.convert_ohlcv_to_mcx(symbol, df)

