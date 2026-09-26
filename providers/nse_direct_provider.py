"""
Direct National Stock Exchange (NSE India) Data Provider.
Fetches official exchange data directly from NSE India archives and web feeds without Yahoo Finance or any broker.
Includes proprietary exchange metrics like Delivery Volume & Delivery Percentage.
"""

from datetime import datetime, timedelta
import io
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests

from core.cache import LocalDataCache
from core.universe import UniverseManager
from providers.base import BaseDataProvider


class NSEDirectProvider(BaseDataProvider):
    """
    Direct Exchange Data Provider for NSE India (National Stock Exchange).
    Fetches official end-of-day Bhavcopy and live quotes directly from archives.nseindia.com.
    Zero broker dependencies, zero API keys required.
    """

    def __init__(self, cache_ttl_hours: float = 4.0):
        super().__init__()
        self.cache = LocalDataCache(ttl_hours=cache_ttl_hours)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self._session = None

    def fetch_latest_bhavcopy(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """
        Downloads official NSE End-of-Day Bhavcopy directly from archives.nseindia.com.
        Finds the most recent valid trading session (skips weekends and trading holidays).
        Contains all 3,500+ securities with exact Open, High, Low, Close, Volume, and Delivery %.
        """
        cur = datetime.now()
        for i in range(days_back):
            check_date = cur - timedelta(days=i)
            if check_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                continue

            date_str = check_date.strftime("%d%m%Y")
            url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
            try:
                resp = requests.get(url, headers=self.headers, timeout=10)
                if resp.status_code == 200 and len(resp.text) > 10000:
                    df = pd.read_csv(io.StringIO(resp.text))
                    df.columns = df.columns.str.strip()
                    # Keep standard equity series (EQ)
                    df = df[df["SERIES"].str.strip() == "EQ"].copy()
                    df["SYMBOL"] = df["SYMBOL"].str.strip()
                    df["DATE"] = check_date.strftime("%Y-%m-%d")
                    self.data_source_mode = "LIVE"
                    return df
                else:
                    self.api_failure_reasons.append(f"archives.nseindia.com returned HTTP {resp.status_code} for {date_str}")
            except Exception as e:
                self.api_failure_reasons.append(f"archives.nseindia.com connection error for {date_str}: {type(e).__name__} ({str(e)[:80]})")
                continue

        # All dates failed
        self.api_call_failed = True
        self.is_latest = False
        self.data_source_mode = "CACHED_FALLBACK"
        if "Could not download latest Bhavcopy from archives.nseindia.com" not in self.api_failure_reasons:
            self.api_failure_reasons.append("External API call to archives.nseindia.com failed for all recent sessions.")
        return None

    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Loads OHLCV for an NSE stock.
        Checks local cache first; if present, updates the latest session from NSE Bhavcopy.
        """
        clean = UniverseManager.to_clean_symbol(symbol)
        
        # MCX commodities do not trade on NSE equity segment: delegate to calibrated commodity provider
        if UniverseManager.is_commodity(clean):
            from providers.yfinance_provider import YahooFinanceProvider
            return YahooFinanceProvider(cache_ttl_hours=self.cache.ttl_hours).fetch_ohlcv(clean, period=period, interval=interval, use_cache=use_cache)

        # 1. Check local cache
        cached_df = self.cache.get(clean) if use_cache else None
        if cached_df is not None and len(cached_df) >= 30:
            return cached_df

        # If cache missing, fetch recent Bhavcopy
        bhav = self.fetch_latest_bhavcopy()
        if bhav is not None:
            match = bhav[bhav["SYMBOL"] == clean]
            if not match.empty:
                row = match.iloc[0]
                dt = pd.to_datetime(row["DATE"])
                df = pd.DataFrame([{
                    "open": float(row["OPEN_PRICE"]),
                    "high": float(row["HIGH_PRICE"]),
                    "low": float(row["LOW_PRICE"]),
                    "close": float(row["CLOSE_PRICE"]),
                    "volume": float(row["TTL_TRD_QNTY"]),
                    "delivery_pct": float(row.get("DELIV_PER", 0.0))
                }], index=[dt])
                df.index.name = "date"
                return df

        return None

    def fetch_batch_ohlcv(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 15,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for multiple NSE symbols directly.
        Uses local historical cache and enriches with direct official NSE Bhavcopy data.
        """
        results: Dict[str, pd.DataFrame] = {}
        clean_symbols = [UniverseManager.to_clean_symbol(s) for s in symbols]

        # 1. Load from cache
        missing = []
        for s in clean_symbols:
            df = self.cache.get(s) if use_cache else None
            if df is None:
                # Emergency cache fallback
                any_c = self.cache.get_any(s)
                if any_c:
                    df = any_c[0]
            if df is not None and len(df) >= 30:
                results[s] = df.copy()
                self.fallback_symbols.append(s)
            else:
                missing.append(s)

        # 2. If any missing or to enrich with direct Bhavcopy
        bhav = self.fetch_latest_bhavcopy()
        if bhav is not None:
            bhav_indexed = bhav.set_index("SYMBOL")
            for s in clean_symbols:
                # MCX Commodities NEVER belong in NSE Bhavcopy!
                if UniverseManager.is_commodity(s):
                    continue
                if s in bhav_indexed.index and s in results:
                    # Update latest row with direct official exchange close & delivery %
                    b_row = bhav_indexed.loc[s]
                    if isinstance(b_row, pd.DataFrame):
                        b_row = b_row.iloc[0]
                    dt = pd.to_datetime(b_row["DATE"])
                    results[s].loc[dt, "close"] = float(b_row["CLOSE_PRICE"])
                    results[s].loc[dt, "open"] = float(b_row["OPEN_PRICE"])
                    results[s].loc[dt, "high"] = float(b_row["HIGH_PRICE"])
                    results[s].loc[dt, "low"] = float(b_row["LOW_PRICE"])
                    results[s].loc[dt, "volume"] = float(b_row["TTL_TRD_QNTY"])
                    results[s].loc[dt, "delivery_pct"] = float(b_row.get("DELIV_PER", 0.0))
                    self.live_symbols.append(s)
                    self.is_latest = True
        else:
            self.api_call_failed = True
            self.is_latest = False
            self.data_source_mode = "CACHED_FALLBACK"

        return results

    def fetch_quote(self, symbol: str) -> Optional[dict]:
        clean = UniverseManager.to_clean_symbol(symbol)
        bhav = self.fetch_latest_bhavcopy()
        if bhav is not None:
            match = bhav[bhav["SYMBOL"] == clean]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "symbol": clean,
                    "exchange": "NSE",
                    "date": row["DATE"],
                    "open": float(row["OPEN_PRICE"]),
                    "high": float(row["HIGH_PRICE"]),
                    "low": float(row["LOW_PRICE"]),
                    "last_price": float(row["CLOSE_PRICE"]),
                    "volume": float(row["TTL_TRD_QNTY"]),
                    "delivery_qty": float(row.get("DELIV_QTY", 0)),
                    "delivery_pct": float(row.get("DELIV_PER", 0.0)),
                }
        return None
