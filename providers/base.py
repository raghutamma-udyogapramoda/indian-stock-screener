"""
Abstract Base Data Provider interface.
Allows swapping between Yahoo Finance, Breeze API, and other data sources seamlessly.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseDataProvider(ABC):
    """Abstract base class for all market data providers."""

    def __init__(self):
        self.api_call_failed: bool = False
        self.is_latest: bool = True
        self.data_source_mode: str = "LIVE"  # "LIVE", "CACHED_FALLBACK", "STATIC_TEST", "API_ERROR"
        self.api_failure_reasons: List[str] = []
        self.failed_symbols: List[str] = []
        self.fallback_symbols: List[str] = []
        self.live_symbols: List[str] = []
        self.symbol_timestamps: Dict[str, str] = {}
        self.symbol_sources: Dict[str, str] = {}
        self.last_sync_time: Optional[str] = None

    def reset_status(self):
        """Resets status telemetry before a new screening batch."""
        self.api_call_failed = False
        self.is_latest = True
        self.data_source_mode = "LIVE"
        self.api_failure_reasons = []
        self.failed_symbols = []
        self.fallback_symbols = []
        self.live_symbols = []
        self.symbol_timestamps = {}
        self.symbol_sources = {}
        self.last_sync_time = None

    def get_health_status(self) -> dict:
        """Returns comprehensive data health and API connectivity status."""
        failed = getattr(self, "failed_symbols", []) or []
        fallback = getattr(self, "fallback_symbols", []) or []
        live = getattr(self, "live_symbols", []) or []
        reasons = getattr(self, "api_failure_reasons", []) or []
        stamps = getattr(self, "symbol_timestamps", {}) or {}
        sources = getattr(self, "symbol_sources", {}) or {}
        api_failed = getattr(self, "api_call_failed", False)
        is_latest_flag = getattr(self, "is_latest", True) and not api_failed

        return {
            "api_call_failed": api_failed,
            "is_latest": is_latest_flag,
            "data_source_mode": getattr(self, "data_source_mode", "LIVE"),
            "failure_reasons": list(dict.fromkeys(reasons)),
            "failed_symbols": sorted(list(set(failed))),
            "fallback_symbols": sorted(list(set(fallback))),
            "live_symbols": sorted(list(set(live))),
            "failed_count": len(set(failed)),
            "fallback_count": len(set(fallback)),
            "live_count": len(set(live)),
            "symbol_timestamps": stamps,
            "symbol_sources": sources,
            "last_sync_time": getattr(self, "last_sync_time", None),
        }


    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data for a single symbol.
        Returned DataFrame MUST have lowercase columns: ['open', 'high', 'low', 'close', 'volume']
        and DatetimeIndex.
        """
        pass

    @abstractmethod
    def fetch_batch_ohlcv(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        max_workers: int = 10,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches historical OHLCV data for multiple symbols concurrently.
        """
        pass

    @abstractmethod
    def fetch_quote(self, symbol: str) -> Optional[dict]:
        """
        Fetches latest quote / snapshot for a symbol.
        """
        pass
