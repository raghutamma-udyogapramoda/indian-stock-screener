"""
Dedicated F&O (Futures & Options) Tracking & Open Interest (OI) Intelligence Engine.
Fetches live derivatives quotes, volume, open interest, and buildup directly from the
National Stock Exchange of India (NSE) official API feeds with resilient multi-tier fallbacks.
Persists user-specific F&O watchlists to disk.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import requests
import yfinance as yf

from config.settings import BASE_DIR
from core.universe import UniverseManager, NSE_FO_SEED, INDICES_SEED

logger = logging.getLogger(__name__)

FNO_TRACKER_DIR = BASE_DIR / "data" / "tracker"
FNO_TRACKER_DIR.mkdir(parents=True, exist_ok=True)

# In-memory short-lived quote cache to minimize NSE request rate
_FNO_QUOTE_CACHE: Dict[str, Tuple[float, dict]] = {}

# Session singleton for NSE requests
_NSE_SESSION: Optional[requests.Session] = None
_NSE_SESSION_WARMED = False
_NSE_SESSION_WARMED_AT = 0.0


def _get_warmed_nse_session() -> requests.Session:
    """
    Initializes and warms up an HTTP session with cookies directly from nseindia.com.
    Re-warms periodically to prevent cookie expiration.
    """
    global _NSE_SESSION, _NSE_SESSION_WARMED, _NSE_SESSION_WARMED_AT

    now = time.time()
    # Re-warm session if older than 10 minutes
    if _NSE_SESSION is None or (not _NSE_SESSION_WARMED) or (now - _NSE_SESSION_WARMED_AT > 600):
        s = requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "*/*",
            "Referer": "https://www.nseindia.com/option-chain",
        }
        s.headers.update(headers)
        try:
            s.get("https://www.nseindia.com", timeout=6)
            s.get("https://www.nseindia.com/option-chain", timeout=6)
            _NSE_SESSION_WARMED = True
            _NSE_SESSION_WARMED_AT = now
        except Exception as e:
            logger.warning("NSE session warmup warning: %s", e)
            _NSE_SESSION_WARMED = False
        _NSE_SESSION = s

    return _NSE_SESSION


class FNOTracker:
    """
    Manages multi-user persistent F&O tracked positions and interfaces
    with official NSE derivatives feeds for real-time Price, Volume & Open Interest.
    """

    @staticmethod
    def _get_user_fno_file(username: str) -> Path:
        clean_user = "".join(c for c in username.lower() if c.isalnum() or c in ("_", "-")).strip()
        if not clean_user:
            clean_user = "default_user"
        return FNO_TRACKER_DIR / f"{clean_user}_fno.json"

    @classmethod
    def load_positions(cls, username: str) -> List[dict]:
        """Loads tracked F&O positions for a specific user from disk."""
        filepath = cls._get_user_fno_file(username)
        if not filepath.exists():
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.error("Failed to load F&O positions for %s: %s", username, e)
            return []

    @classmethod
    def save_positions(cls, username: str, positions: List[dict]) -> None:
        """Saves tracked F&O positions for a specific user to disk."""
        filepath = cls._get_user_fno_file(username)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(positions, f, indent=2, default=str)
        except Exception as e:
            logger.error("Error saving F&O portfolio for %s: %s", username, e)

    @classmethod
    def add_position(
        cls,
        username: str,
        symbol: str,
        contract_type: str = "FUTURES",  # FUTURES, OPTION_CE, OPTION_PE
        strike: float = 0.0,
        expiry: str = "",
        lot_size: Optional[int] = None,
        lots: int = 1,
        buy_price: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        notes: str = ""
    ) -> dict:
        """Adds a new derivative contract / stock to user's F&O tracker."""
        positions = cls.load_positions(username)
        clean_symbol = cls.clean_symbol(symbol)

        # Resolve standard lot size if not specified
        resolved_lot_size = lot_size or cls.get_default_lot_size(clean_symbol)

        # Generate unique position ID
        pos_id = f"fno_{clean_symbol}_{int(time.time() * 1000)}"

        # If buy price is 0, attempt to auto-fetch live CMP
        resolved_buy = float(buy_price)
        if resolved_buy <= 0:
            quote = cls.fetch_fno_quote(clean_symbol, use_cache=True)
            if contract_type == "FUTURES":
                resolved_buy = quote.get("fut_price") or quote.get("spot_cmp", 100.0)
            elif "OPTION" in contract_type:
                # Find matching strike in option chain if available
                opt_price = 0.0
                if strike > 0 and quote.get("options_data"):
                    target_type = "CE" if "CE" in contract_type else "PE"
                    for opt in quote.get("options_data", []):
                        if opt.get("optionType") == target_type and abs(float(opt.get("strikePrice", 0)) - strike) < 0.1:
                            opt_price = float(opt.get("lastPrice", 0.0))
                            break
                resolved_buy = opt_price if opt_price > 0 else (quote.get("spot_cmp", 100.0) * 0.02)
            else:
                resolved_buy = quote.get("spot_cmp", 100.0)

        # Default stop loss & target if 0
        resolved_sl = float(stop_loss)
        resolved_tgt = float(target)
        if resolved_sl <= 0 and resolved_buy > 0:
            resolved_sl = round(resolved_buy * (0.97 if "PE" not in contract_type else 1.03), 2)
        if resolved_tgt <= 0 and resolved_buy > 0:
            resolved_tgt = round(resolved_buy * (1.06 if "PE" not in contract_type else 0.94), 2)

        new_pos = {
            "id": pos_id,
            "symbol": clean_symbol,
            "contract_type": contract_type.upper(),
            "strike": float(strike),
            "expiry": expiry.strip(),
            "lot_size": int(resolved_lot_size),
            "lots": int(lots),
            "total_qty": int(resolved_lot_size * lots),
            "buy_price": round(resolved_buy, 2),
            "stop_loss": round(resolved_sl, 2),
            "target": round(resolved_tgt, 2),
            "notes": notes.strip(),
            "added_at": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%d %H:%M:%S")
        }

        # Avoid duplicates for same symbol + contract_type + strike + expiry
        positions = [
            p for p in positions
            if not (
                p.get("symbol") == clean_symbol
                and p.get("contract_type") == contract_type.upper()
                and abs(float(p.get("strike", 0)) - float(strike)) < 0.1
                and p.get("expiry") == expiry.strip()
            )
        ]
        positions.insert(0, new_pos)
        cls.save_positions(username, positions)
        return new_pos

    @classmethod
    def add_watchlist_item(
        cls,
        username: str,
        symbol: str,
        contract_type: str = "FUTURES",
        strike_price: float = 0.0,
        expiry_date: str = "",
        lot_size: Optional[int] = None,
        lots: int = 1,
        buy_price: float = 0.0,
        stop_loss: float = 0.0,
        target: float = 0.0,
        notes: str = "",
        **kwargs
    ) -> dict:
        """
        Alias / compatibility wrapper for add_position to support watchlist tracking semantics.
        Accepts both strike_price/expiry_date and strike/expiry keyword arguments.
        """
        resolved_strike = float(kwargs.get("strike", strike_price or 0.0))
        resolved_expiry = str(kwargs.get("expiry", expiry_date or "")).strip()
        return cls.add_position(
            username=username,
            symbol=symbol,
            contract_type=contract_type,
            strike=resolved_strike,
            expiry=resolved_expiry,
            lot_size=lot_size,
            lots=lots,
            buy_price=buy_price,
            stop_loss=stop_loss,
            target=target,
            notes=notes
        )

    @classmethod
    def delete_position(cls, username: str, position_id: str) -> bool:
        """Deletes a tracked F&O position by ID."""
        positions = cls.load_positions(username)
        initial_len = len(positions)
        positions = [p for p in positions if p.get("id") != position_id]
        if len(positions) < initial_len:
            cls.save_positions(username, positions)
            return True
        return False

    @classmethod
    def clear_all_positions(cls, username: str) -> bool:
        """Removes all F&O positions for a user."""
        cls.save_positions(username, [])
        return True

    @staticmethod
    def clean_symbol(symbol: str) -> str:
        """Extracts standard underlying equity / index symbol."""
        if not symbol:
            return ""
        s = symbol.strip().upper()
        # Clean common prefixes/suffixes
        s = s.replace(".NS", "").replace(".BO", "").replace("^", "")
        # Remove trailing option strikes like 'RELIANCE 1200 CE'
        tokens = s.replace("-", " ").split()
        if tokens:
            return tokens[0]
        return s

    @staticmethod
    def get_default_lot_size(symbol: str) -> int:
        """Returns standard lot size for Indian Indices & F&O stocks."""
        sym = symbol.strip().upper()
        # Indices
        index_lots = {
            "NIFTY": 65,
            "BANKNIFTY": 30,
            "SENSEX": 20,
            "FINNIFTY": 60,
            "MIDCPNIFTY": 120,
            "NIFTYIT": 25,
            "BANKEX": 30
        }
        if sym in index_lots:
            return index_lots[sym]

        # Common high-volume stock lot sizes
        stock_lots = {
            "RELIANCE": 250,
            "TCS": 175,
            "INFY": 400,
            "HDFCBANK": 550,
            "ICICIBANK": 700,
            "SBIN": 750,
            "BHARTIARTL": 475,
            "AXISBANK": 625,
            "KOTAKBANK": 400,
            "LT": 150,
            "ITC": 1600,
            "TATASTEEL": 5500,
            "MARUTI": 50,
            "BAJFINANCE": 125,
            "TATAMOTORS": 1425,
            "TMPV": 1425,
            "TRENT": 100,
            "BEL": 2850,
            "COALINDIA": 2100,
            "NTPC": 1500,
            "POWERGRID": 1800,
            "HCLTECH": 350,
            "WIPRO": 1500,
            "SUNPHARMA": 350,
            "ONGC": 3850
        }
        return stock_lots.get(sym, 250)

    @classmethod
    def fetch_fno_quote(cls, symbol: str, use_cache: bool = True, cache_ttl_sec: float = 30.0) -> dict:
        """
        Fetches live derivative data from NSE India website.
        Extracts Spot CMP, Futures Price, Basis, Total Volume, Open Interest (OI),
        OI Change %, Buildup interpretation, Put-Call Ratio (PCR), and major strikes.
        Gracefully falls back to Yahoo Finance if NSE connection is unavailable.
        """
        clean_sym = cls.clean_symbol(symbol)
        now = time.time()

        # Check in-memory cache
        if use_cache and clean_sym in _FNO_QUOTE_CACHE:
            cached_time, cached_data = _FNO_QUOTE_CACHE[clean_sym]
            if now - cached_time < cache_ttl_sec:
                return cached_data

        # 1. Attempt official NSE Derivatives API
        try:
            quote_data = cls._fetch_from_nse_direct(clean_sym)
            if quote_data and not quote_data.get("is_fallback"):
                _FNO_QUOTE_CACHE[clean_sym] = (now, quote_data)
                return quote_data
        except Exception as e:
            logger.warning("NSE Direct derivative fetch failed for %s: %s", clean_sym, e)

        # 2. Resilient Fallback using Yahoo Finance + Synthetic OI Estimator
        fallback_data = cls._fetch_fallback_quote(clean_sym)
        _FNO_QUOTE_CACHE[clean_sym] = (now, fallback_data)
        return fallback_data

    @classmethod
    def _fetch_from_nse_direct(cls, symbol: str) -> Optional[dict]:
        """Queries the official NSE India apiClient NextApi endpoint."""
        session = _get_warmed_nse_session()
        url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolDerivativesData&symbol={symbol}"

        try:
            resp = session.get(url, timeout=7)
            if resp.status_code != 200:
                return None

            payload = resp.json()
            data = payload.get("data", [])
            if not data or not isinstance(data, list):
                return None

            # Spot price
            spot_cmp = 0.0
            for item in data:
                if item.get("underlyingValue"):
                    try:
                        spot_cmp = float(item["underlyingValue"])
                        if spot_cmp > 0:
                            break
                    except (ValueError, TypeError):
                        pass

            # Futures contracts
            futs = [x for x in data if "FUT" in str(x.get("instrumentType", ""))]
            near_fut = futs[0] if futs else {}

            fut_price = float(near_fut.get("lastPrice", spot_cmp or 0.0))
            if spot_cmp == 0.0 and fut_price > 0.0:
                spot_cmp = fut_price

            basis = fut_price - spot_cmp
            basis_pct = (basis / spot_cmp * 100.0) if spot_cmp > 0 else 0.0

            fut_change = float(near_fut.get("change", 0.0))
            fut_pchange = float(near_fut.get("pchange", 0.0))
            fut_oi = int(near_fut.get("openInterest", 0))
            fut_oi_change = int(near_fut.get("changeinOpenInterest", 0))
            fut_oi_pchange = float(near_fut.get("pchangeinOpenInterest", 0.0))
            fut_expiry = str(near_fut.get("expiryDate", ""))

            # Total Open Interest across all futures months
            total_fut_oi = sum(int(x.get("openInterest", 0)) for x in futs)
            total_fut_oi_change = sum(int(x.get("changeinOpenInterest", 0)) for x in futs)

            # Institutional Buildup Classification
            buildup_dict = cls.compute_oi_buildup(fut_pchange, fut_oi_pchange)

            # Options contracts
            ce_items = [x for x in data if str(x.get("optionType", "")).upper() == "CE"]
            pe_items = [x for x in data if str(x.get("optionType", "")).upper() == "PE"]

            total_ce_oi = sum(int(x.get("openInterest", 0)) for x in ce_items)
            total_pe_oi = sum(int(x.get("openInterest", 0)) for x in pe_items)
            pcr = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 1.0

            # Find Major Resistance (Highest CE OI) and Major Support (Highest PE OI)
            highest_ce = max(ce_items, key=lambda x: int(x.get("openInterest", 0))) if ce_items else {}
            highest_pe = max(pe_items, key=lambda x: int(x.get("openInterest", 0))) if pe_items else {}

            res_strike = 0.0
            sup_strike = 0.0
            try:
                res_strike = float(str(highest_ce.get("strikePrice", "0")).strip())
            except Exception:
                pass
            try:
                sup_strike = float(str(highest_pe.get("strikePrice", "0")).strip())
            except Exception:
                pass

            # Expiry dates list
            expiries = []
            seen_exp = set()
            for x in data:
                e = str(x.get("expiryDate", "")).strip()
                if e and e not in seen_exp:
                    seen_exp.add(e)
                    expiries.append(e)

            # Volume & Turnover
            total_contracts = sum(int(x.get("totalTradedVolume", 0)) for x in data)
            total_turnover_cr = round(sum(float(x.get("totalTurnover", 0)) for x in data) / 1e7, 2)

            timestamp = payload.get("timestamp", datetime.now().strftime("%d-%b-%Y %H:%M:%S"))

            return {
                "symbol": symbol,
                "spot_cmp": round(spot_cmp, 2),
                "fut_price": round(fut_price, 2),
                "basis": round(basis, 2),
                "basis_pct": round(basis_pct, 2),
                "basis_type": "PREMIUM" if basis >= 0 else "DISCOUNT",
                "change": round(fut_change, 2),
                "change_pct": round(fut_pchange, 2),
                "near_fut_oi": fut_oi,
                "near_fut_oi_change": fut_oi_change,
                "near_fut_oi_pchange": round(fut_oi_pchange, 2),
                "total_fut_oi": total_fut_oi,
                "total_fut_oi_change": total_fut_oi_change,
                "fut_expiry": fut_expiry,
                "buildup": buildup_dict["code"],
                "buildup_badge": buildup_dict["badge"],
                "buildup_desc": buildup_dict["description"],
                "buildup_color": buildup_dict["color"],
                "buildup_sentiment": buildup_dict["sentiment"],
                "total_ce_oi": total_ce_oi,
                "total_pe_oi": total_pe_oi,
                "pcr": pcr,
                "resistance_strike": res_strike,
                "support_strike": sup_strike,
                "total_volume": total_contracts,
                "total_turnover_cr": total_turnover_cr,
                "expiry_dates": expiries[:5],
                "options_data": data[:60],  # sample for strike lookups
                "source": "NSE_OFFICIAL_LIVE",
                "is_fallback": False,
                "timestamp": timestamp
            }
        except Exception as e:
            logger.warning("Error parsing NSE derivatives response: %s", e)
            return None

    @classmethod
    def _fetch_fallback_quote(cls, symbol: str) -> dict:
        """
        Robust backup quote using Yahoo Finance when NSE website is unreachable or rate-limited.
        Synthesizes technical derivatives metrics so UI never shows blank or error states.
        """
        yf_sym = UniverseManager.to_yfinance_symbol(symbol)
        cmp_val = 0.0
        change_pct = 0.0
        day_vol = 0
        prev_close = 0.0

        try:
            ticker = yf.Ticker(yf_sym)
            fi = ticker.fast_info
            cmp_val = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None) or 0.0
            prev_close = getattr(fi, "previous_close", None) or cmp_val
            day_vol = getattr(fi, "last_volume", 0) or 0
            if prev_close > 0 and cmp_val > 0:
                change_pct = round(((cmp_val - prev_close) / prev_close) * 100.0, 2)
        except Exception:
            cmp_val = 100.0

        # Estimated synthetic futures and OI metrics
        basis = round(cmp_val * 0.002, 2)  # slight typical 0.2% premium
        fut_price = round(cmp_val + basis, 2)
        synthetic_oi_chg = round(change_pct * 1.5, 2)
        buildup_dict = cls.compute_oi_buildup(change_pct, synthetic_oi_chg)

        # Standard step interval for strikes
        step = 50.0 if cmp_val > 1000 else (20.0 if cmp_val > 500 else 10.0)
        res_strike = round((cmp_val * 1.03) / step) * step
        sup_strike = round((cmp_val * 0.97) / step) * step

        ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d-%b-%Y %H:%M:%S IST")

        return {
            "symbol": symbol,
            "spot_cmp": round(cmp_val, 2),
            "fut_price": round(fut_price, 2),
            "basis": basis,
            "basis_pct": 0.20,
            "basis_type": "PREMIUM",
            "change": round(cmp_val - prev_close, 2),
            "change_pct": change_pct,
            "near_fut_oi": 50000,
            "near_fut_oi_change": int(synthetic_oi_chg * 500),
            "near_fut_oi_pchange": synthetic_oi_chg,
            "total_fut_oi": 75000,
            "total_fut_oi_change": int(synthetic_oi_chg * 750),
            "fut_expiry": "Near-Month",
            "buildup": buildup_dict["code"],
            "buildup_badge": buildup_dict["badge"],
            "buildup_desc": buildup_dict["description"],
            "buildup_color": buildup_dict["color"],
            "buildup_sentiment": buildup_dict["sentiment"],
            "total_ce_oi": 150000,
            "total_pe_oi": 140000,
            "pcr": 0.93,
            "resistance_strike": res_strike,
            "support_strike": sup_strike,
            "total_volume": day_vol or 10000,
            "total_turnover_cr": round((day_vol * cmp_val) / 1e7, 2) if day_vol else 50.0,
            "expiry_dates": ["Current Expiry", "Next Month"],
            "options_data": [],
            "source": "FALLBACK_HYBRID",
            "is_fallback": True,
            "timestamp": ist_now
        }

    @staticmethod
    def compute_oi_buildup(price_change_pct: float, oi_change_pct: float) -> dict:
        """
        Classifies Open Interest (OI) buildup into standard institutional dynamics:
          - Price UP & OI UP   -> Long Buildup (Aggressive Buying)
          - Price DOWN & OI UP -> Short Buildup (Aggressive Shorting)
          - Price UP & OI DOWN -> Short Covering (Shorts Squeezed Out / Covering)
          - Price DOWN & OI DOWN -> Long Unwinding (Longs Liquidating / Exiting)
        """
        p_up = price_change_pct >= 0.0
        oi_up = oi_change_pct >= 0.0

        if p_up and oi_up:
            return {
                "code": "LONG_BUILDUP",
                "badge": "🔥 Long Buildup",
                "sentiment": "BULLISH",
                "description": "Price Rising with Fresh OI Expansion: Strong institutional accumulation and bullish commitment.",
                "color": "#00E676"
            }
        elif (not p_up) and oi_up:
            return {
                "code": "SHORT_BUILDUP",
                "badge": "📉 Short Buildup",
                "sentiment": "BEARISH",
                "description": "Price Falling with Fresh OI Expansion: Aggressive institutional short positioning.",
                "color": "#FF5252"
            }
        elif p_up and (not oi_up):
            return {
                "code": "SHORT_COVERING",
                "badge": "⚡ Short Covering",
                "sentiment": "BULLISH (SQUEEZE)",
                "description": "Price Rising with OI Contraction: Trapped bears exiting short positions, driving upside squeeze.",
                "color": "#38BDF8"
            }
        else:
            return {
                "code": "LONG_UNWINDING",
                "badge": "⚠️ Long Unwinding",
                "sentiment": "BEARISH (WEAKNESS)",
                "description": "Price Falling with OI Contraction: Bulls dumping long positions, leading to downward drift.",
                "color": "#F59E0B"
            }

    @classmethod
    def fetch_batch_fno_quotes(cls, symbols: List[str], use_cache: bool = True) -> Dict[str, dict]:
        """Parallel multi-threaded fetcher for all tracked F&O symbols."""
        if not symbols:
            return {}

        results: Dict[str, dict] = {}
        unique_syms = list(dict.fromkeys(cls.clean_symbol(s) for s in symbols if s))

        def _fetch_worker(sym):
            return sym, cls.fetch_fno_quote(sym, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=min(len(unique_syms), 8)) as executor:
            for s, q in executor.map(lambda x: _fetch_worker(x), unique_syms):
                results[s] = q

        return results

    @classmethod
    def get_all_searchable_fno_symbols(cls) -> List[str]:
        """Returns sorted list of all active F&O underlying symbols & indices."""
        fno_list = sorted(list(set(INDICES_SEED + NSE_FO_SEED)))
        return fno_list
