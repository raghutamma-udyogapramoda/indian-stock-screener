"""
Abstract Base Screener class.
Provides standardized interface for all algorithmic filtering strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseScreener(ABC):
    """Base class for strategy screeners."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def screen(self, symbol: str, df: pd.DataFrame, benchmark_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[dict]:
        """
        Evaluates a single stock DataFrame against the quantitative criteria.
        Returns a candidate summary dict if qualified, else None.
        """
        pass

    def screen_batch(
        self,
        data: Dict[str, pd.DataFrame],
        top_n: int = 10,
        benchmark_data: Optional[Dict[str, pd.DataFrame]] = None,
        **kwargs
    ) -> List[dict]:
        """
        Screens a dictionary of {symbol: df} and returns top candidates sorted by score.
        Automatically utilizes benchmark_data for sector tailwind and panic-day resilience.
        """
        if benchmark_data is None:
            # Auto-extract any indices already present in data
            benchmark_data = {
                s: d for s, d in data.items()
                if s in ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT", "^NSEI", "^NSEBANK", "^BSESN", "^CNXIT"]
            }

        candidates = []
        for symbol, df in data.items():
            # Skip indices from stock screeners
            if symbol in ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT", "INDIAVIX", "^NSEI", "^NSEBANK", "^BSESN", "^CNXIT", "^INDIAVIX"]:
                continue
            def _enrich_candidate(c: dict, d: pd.DataFrame) -> dict:
                if hasattr(d, "attrs"):
                    if "capture_time" in d.attrs and "capture_time" not in c:
                        c["capture_time"] = d.attrs["capture_time"]
                    if "data_source" in d.attrs and "data_source" not in c:
                        c["data_source"] = d.attrs["data_source"]
                if "candle_date" not in c and d is not None and not d.empty and isinstance(d.index, pd.DatetimeIndex):
                    last_idx = d.index[-1]
                    c["candle_date"] = last_idx.strftime("%d-%b-%Y") if hasattr(last_idx, "strftime") else str(last_idx)[:10]
                return c

            try:
                candidate = self.screen(symbol, df, benchmark_data=benchmark_data)
                if candidate:
                    candidates.append(_enrich_candidate(candidate, df))
            except TypeError:
                try:
                    candidate = self.screen(symbol, df)
                    if candidate:
                        candidates.append(_enrich_candidate(candidate, df))
                except Exception:
                    continue
            except Exception:
                continue

        # Sort by quantitative quality score descending
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        return candidates[:top_n]

