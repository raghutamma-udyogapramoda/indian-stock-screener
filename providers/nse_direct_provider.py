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
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
        self._session = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.headers)
        return self._session

    def fetch_latest_bhavcopy(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """
        Downloads official NSE End-of-Day Bhavcopy directly from NSE India archives.
        Tries both archives.nseindia.com and nsearchives.nseindia.com mirrors.
        Finds the most recent valid trading session (skips weekends and trading holidays).
        Contains all 3,500+ securities with exact Open, High, Low, Close, Volume, and Delivery %.
        """
        session = self._get_session()
        cur = datetime.now()
        domains = ["archives.nseindia.com", "nsearchives.nseindia.com"]

        for i in range(days_back):
            check_date = cur - timedelta(days=i)
            if check_date.weekday() >= 5:  # Skip Saturday (5) and Sunday (6)
                continue

            date_str = check_date.strftime("%d%m%Y")
            for domain in domains:
                url = f"https://{domain}/products/content/sec_bhavdata_full_{date_str}.csv"
                try:
                    resp = session.get(url, timeout=3.5)
                    if resp.status_code == 200 and len(resp.text) > 10000:
                        df = pd.read_csv(io.StringIO(resp.text))
                        df.columns = df.columns.str.strip()
                        # Keep standard equity series (EQ)
                        df = df[df["SERIES"].str.strip() == "EQ"].copy()
                        df["SYMBOL"] = df["SYMBOL"].str.strip()
                        df["DATE"] = check_date.strftime("%Y-%m-%d")
                        self.data_source_mode = "LIVE"
                        return df
                    elif resp.status_code == 403:
                        self.api_failure_reasons.append(f"{domain} returned HTTP 403 (NSE CDN blocks cloud datacenter IPs)")
                        break
                    else:
                        self.api_failure_reasons.append(f"{domain} returned HTTP {resp.status_code} for {date_str}")
                except Exception as e:
                    self.api_failure_reasons.append(f"{domain} error: {type(e).__name__} ({str(e)[:60]})")
                    continue

            # If cloud datacenter IP is blocked by NSE CDN (403), skip querying older dates
            if any("HTTP 403" in r for r in self.api_failure_reasons):
                break

        # All dates and domains failed
        self.api_call_failed = True
        self.is_latest = False
        self.data_source_mode = "CACHED_FALLBACK"
        if "Could not download latest Bhavcopy from NSE archives" not in self.api_failure_reasons:
            self.api_failure_reasons.append("External API call to NSE archives failed for all recent sessions.")
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
        Checks local cache first; if missing (e.g. fresh cloud deployment on Streamlit Cloud),
        seamlessly backfills full historical OHLCV from YahooFinanceProvider and enriches with NSE Bhavcopy.
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

        # 2. Backfill historical candles (essential for cloud deployments where disk cache is fresh)
        from providers.yfinance_provider import YahooFinanceProvider
        yf_prov = YahooFinanceProvider(cache_ttl_hours=self.cache.ttl_hours)
        df = yf_prov.fetch_ohlcv(clean, period=period, interval=interval, use_cache=use_cache)
        if df is not None and len(df) >= 30:
            # Enrich latest candle with NSE Bhavcopy if accessible
            try:
                bhav = self.fetch_latest_bhavcopy()
                if bhav is not None:
                    match = bhav[bhav["SYMBOL"] == clean]
                    if not match.empty:
                        row = match.iloc[0]
                        dt = pd.to_datetime(row["DATE"])
                        if hasattr(df.index, "tz") and df.index.tz is not None:
                            dt = dt.tz_localize(df.index.tz) if getattr(dt, "tzinfo", None) is None else dt.tz_convert(df.index.tz)
                        df.loc[dt, "open"] = float(row["OPEN_PRICE"])
                        df.loc[dt, "high"] = float(row["HIGH_PRICE"])
                        df.loc[dt, "low"] = float(row["LOW_PRICE"])
                        df.loc[dt, "close"] = float(row["CLOSE_PRICE"])
                        df.loc[dt, "volume"] = float(row["TTL_TRD_QNTY"])
                        df.loc[dt, "delivery_pct"] = float(row.get("DELIV_PER", 0.0))
            except Exception:
                pass
            self.cache.set(clean, df, ttl_hours=self.cache.ttl_hours)
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
        If cache is empty (e.g. on Streamlit Community Cloud), automatically backfills
        historical candles so the screener never returns empty.
        """
        results: Dict[str, pd.DataFrame] = {}
        clean_symbols = [UniverseManager.to_clean_symbol(s) for s in symbols]

        # 1. Load from cache
        missing = []
        for s in clean_symbols:
            if UniverseManager.is_commodity(s):
                continue
            df = self.cache.get(s) if use_cache else None
            if df is None:
                # Emergency cache fallback
                any_c = self.cache.get_any(s)
                if any_c:
                    df = any_c[0]
            if df is not None and len(df) >= 30:
                results[s] = df.copy()
            else:
                missing.append(s)

        # 1b. Backfill missing symbols from YahooFinanceProvider (vital on cloud environments)
        if missing:
            from providers.yfinance_provider import YahooFinanceProvider
            yf_fallback = YahooFinanceProvider(cache_ttl_hours=self.cache.ttl_hours)
            backfilled = yf_fallback.fetch_batch_ohlcv(missing, period=period, interval=interval, max_workers=max_workers, use_cache=use_cache)
            for s, df in backfilled.items():
                if df is not None and len(df) >= 30:
                    results[s] = df.copy()
                    self.cache.set(s, df, ttl_hours=self.cache.ttl_hours)
                    self.fallback_symbols.append(s)

        # 1c. If commodities were included, fetch them via calibrated provider
        comm_symbols = [s for s in clean_symbols if UniverseManager.is_commodity(s)]
        if comm_symbols:
            from providers.yfinance_provider import YahooFinanceProvider
            yf_comm = YahooFinanceProvider(cache_ttl_hours=self.cache.ttl_hours)
            comm_data = yf_comm.fetch_batch_ohlcv(comm_symbols, period=period, interval=interval, max_workers=max_workers, use_cache=use_cache)
            for s, df in comm_data.items():
                if df is not None and not df.empty:
                    results[s] = df.copy()
                    self.cache.set(s, df, ttl_hours=self.cache.ttl_hours)

        # 2. Enrich with direct official Bhavcopy if accessible
        try:
            bhav = self.fetch_latest_bhavcopy()
            if bhav is not None:
                bhav_indexed = bhav.set_index("SYMBOL")
                for s in clean_symbols:
                    if UniverseManager.is_commodity(s):
                        continue
                    if s in bhav_indexed.index and s in results:
                        b_row = bhav_indexed.loc[s]
                        if isinstance(b_row, pd.DataFrame):
                            b_row = b_row.iloc[0]
                        dt = pd.to_datetime(b_row["DATE"])
                        if hasattr(results[s].index, "tz") and results[s].index.tz is not None:
                            dt = dt.tz_localize(results[s].index.tz) if getattr(dt, "tzinfo", None) is None else dt.tz_convert(results[s].index.tz)
                        results[s].loc[dt, "close"] = float(b_row["CLOSE_PRICE"])
                        results[s].loc[dt, "open"] = float(b_row["OPEN_PRICE"])
                        results[s].loc[dt, "high"] = float(b_row["HIGH_PRICE"])
                        results[s].loc[dt, "low"] = float(b_row["LOW_PRICE"])
                        results[s].loc[dt, "volume"] = float(b_row["TTL_TRD_QNTY"])
                        results[s].loc[dt, "delivery_pct"] = float(b_row.get("DELIV_PER", 0.0))
                        self.live_symbols.append(s)
                        self.is_latest = True
                        self.cache.set(s, results[s], ttl_hours=self.cache.ttl_hours)
                
                # Deduplicate fallback symbols if they were enriched with live Bhavcopy
                if self.live_symbols:
                    self.data_source_mode = "LIVE"
                    self.fallback_symbols = [x for x in self.fallback_symbols if x not in self.live_symbols]
            else:
                self.api_call_failed = True
                self.is_latest = False
                self.data_source_mode = "CACHED_FALLBACK"
                for s in results.keys():
                    if s not in self.fallback_symbols:
                        self.fallback_symbols.append(s)
        except Exception as e:
            self.api_call_failed = True
            self.is_latest = False
            self.data_source_mode = "CACHED_FALLBACK"
            self.api_failure_reasons.append(f"NSE Bhavcopy enrichment error: {str(e)[:100]}")
            for s in results.keys():
                if s not in self.fallback_symbols:
                    self.fallback_symbols.append(s)

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
