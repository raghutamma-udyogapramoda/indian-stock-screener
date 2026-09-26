"""
Local Parquet-based Cache for fast historical and daily candle loading.
Enables instant screening and avoids redundant network API calls when scaling to 2000+ stocks.
"""

import time
from pathlib import Path
from typing import Optional
import pandas as pd

from config.settings import CACHE_DIR


class LocalDataCache:
    """Manages reading and writing ticker historical data to disk in Parquet format."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: float = 4.0):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, symbol: str) -> Path:
        clean = symbol.upper().replace(".NS", "").replace(".BO", "").replace("^", "")
        return self.cache_dir / f"{clean}.parquet"

    def get(self, symbol: str, max_age_hours: Optional[float] = None) -> Optional[pd.DataFrame]:
        """Loads cached DataFrame if it exists and is not expired."""
        path = self._get_path(symbol)
        if not path.exists():
            return None

        # Check TTL
        file_age = time.time() - path.stat().st_mtime
        allowed_age = (max_age_hours * 3600) if max_age_hours is not None else self.ttl_seconds
        if file_age > allowed_age:
            return None

        try:
            df = pd.read_parquet(path)
            return df
        except Exception:
            return None

    def get_any(self, symbol: str) -> Optional[tuple[pd.DataFrame, float]]:
        """Loads cached DataFrame regardless of age, returning (df, age_hours). Used for emergency fallback when external API fails."""
        path = self._get_path(symbol)
        if not path.exists():
            return None
        try:
            age_hours = (time.time() - path.stat().st_mtime) / 3600.0
            df = pd.read_parquet(path)
            return df, age_hours
        except Exception:
            return None

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Saves DataFrame to local Parquet cache."""
        if df is None or df.empty:
            return
        path = self._get_path(symbol)
        try:
            df.to_parquet(path, engine="pyarrow")
        except Exception:
            # Fallback to csv if pyarrow has any transient issue
            try:
                df.to_csv(path.with_suffix(".csv"))
            except Exception:
                pass

    def exists(self, symbol: str) -> bool:
        return self._get_path(symbol).exists()
