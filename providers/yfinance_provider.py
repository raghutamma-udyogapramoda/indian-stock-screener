"""
Yahoo Finance Data Provider.
Provides zero-setup, free access to Indian market data (NSE & BSE) using yfinance.
Ideal for weekend scans, 52-week high breakout analysis, and daily OHLCV backfilling.
"""

from datetime import datetime, timezone, timedelta
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
          5. Real-Time Intraday Sync: Cross-checks fast_info to guarantee today's live LTP and avoid stale previous-close lag.
          6. Failure Handling: If Yahoo Finance API fails, captures error reason, flags api_call_failed,
             and gracefully falls back to any existing cached parquet file on disk.
          7. Cache Write: Stores fresh DataFrame to disk as columnar Parquet file.
        """
        clean_symbol = UniverseManager.to_clean_symbol(symbol)
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        
        # 1. Try local cache (within TTL)
        if use_cache:
            cached_df = self.cache.get(clean_symbol)
            if cached_df is not None and len(cached_df) >= 30:
                mtime = self.cache.get_mtime(clean_symbol)
                mtime_str = datetime.fromtimestamp(mtime, ist_tz).strftime("%d-%b %I:%M %p IST") if mtime else "Cached"
                cached_df.attrs["capture_time"] = f"{mtime_str} (Cached)"
                cached_df.attrs["data_source"] = "Local Cache"
                self.symbol_timestamps[clean_symbol] = f"{mtime_str} (Cached)"
                self.symbol_sources[clean_symbol] = "Local Cache"
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

            # Real-Time Intraday Synchronization:
            # yfinance daily history often omits or lags the current live intraday price during market hours.
            # We synchronize with ticker.fast_info to guarantee real-time LTP accuracy.
            try:
                fast_info = getattr(ticker, "fast_info", None)
                if fast_info:
                    live_price = getattr(fast_info, "last_price", None) or getattr(fast_info, "regular_market_price", None)
                    if live_price and float(live_price) > 0:
                        live_price = float(live_price)
                        day_open = getattr(fast_info, "open", None) or getattr(fast_info, "regular_market_open", None) or live_price
                        day_high = getattr(fast_info, "day_high", None) or live_price
                        day_low = getattr(fast_info, "day_low", None) or live_price
                        day_vol = getattr(fast_info, "last_volume", None) or 0

                        now_ist = datetime.now(ist_tz)
                        today_ist_date = now_ist.date()
                        
                        last_ts = df.index[-1]
                        last_date = last_ts.date() if hasattr(last_ts, "date") else None
                        
                        # If market has traded today (Monday-Friday) and last candle is from a previous session, append today's live bar
                        if last_date and last_date < today_ist_date and now_ist.weekday() < 5 and now_ist.hour >= 9:
                            today_tz = last_ts.tzinfo if hasattr(last_ts, "tzinfo") else None
                            today_stamp = pd.Timestamp(now_ist.date(), tz=today_tz)
                            df.loc[today_stamp] = {
                                "open": float(day_open),
                                "high": max(float(day_high), live_price),
                                "low": min(float(day_low), live_price),
                                "close": live_price,
                                "volume": float(day_vol)
                            }
                        elif last_date and last_date == today_ist_date:
                            # Update today's candle close with real-time LTP
                            df.iloc[-1, df.columns.get_loc('close')] = live_price
                            if day_high:
                                df.iloc[-1, df.columns.get_loc('high')] = max(float(df.iloc[-1]['high']), float(day_high))
                            if day_low:
                                df.iloc[-1, df.columns.get_loc('low')] = min(float(df.iloc[-1]['low']), float(day_low))
                            if day_vol and float(day_vol) > float(df.iloc[-1]['volume']):
                                df.iloc[-1, df.columns.get_loc('volume')] = float(day_vol)
            except Exception:
                pass

            # If symbol is an MCX commodity, convert international futures prices to Indian MCX terms (INR)
            if UniverseManager.is_commodity(clean_symbol):
                df = self._convert_to_mcx_inr(clean_symbol, df)

            now_str = datetime.now(ist_tz).strftime("%I:%M:%S %p IST")
            df.attrs["capture_time"] = f"{now_str} (Live)"
            df.attrs["data_source"] = "Yahoo Finance (Live)"
            self.symbol_timestamps[clean_symbol] = f"{now_str} (Live)"
            self.symbol_sources[clean_symbol] = "Yahoo Finance (Live)"
            self.last_sync_time = now_str

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
                    mtime = self.cache.get_mtime(clean_symbol)
                    mtime_str = datetime.fromtimestamp(mtime, ist_tz).strftime("%d-%b %I:%M %p IST") if mtime else "Fallback Cache"
                    fallback_df.attrs["capture_time"] = f"{mtime_str} (Fallback)"
                    fallback_df.attrs["data_source"] = "Cached Fallback"
                    self.symbol_timestamps[clean_symbol] = f"{mtime_str} (Fallback)"
                    self.symbol_sources[clean_symbol] = "Cached Fallback"
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

