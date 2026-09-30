"""
Universe Manager for Indian Equities (NSE/BSE).
Supports Nifty 50, Nifty 500, F&O Universe, and Custom/All Stocks.
"""

import os
from pathlib import Path
from typing import List, Optional
import time
import pandas as pd
import requests
import yfinance as yf

from config.settings import UNIVERSES_DIR

_USDINR_CACHE = {
    "rate": 95.5,
    "timestamp": 0.0
}

# Reliable built-in seed list for Nifty 50
NIFTY_50_SEED = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL", "SBIN", "LICI",
    "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH", "MARUTI", "SUNPHARMA",
    "ONGC", "KOTAKBANK", "TITAN", "TMPV", "NTPC", "AXISBANK", "ADANIENT",
    "COALINDIA", "POWERGRID", "ASIANPAINT", "ULTRACEMCO", "BAJAJFINSV", "M&M",
    "TATASTEEL", "SIEMENS", "GRASIM", "TECHM", "NESTLEIND", "JSWSTEEL", "ADANIPORTS",
    "SBILIFE", "WIPRO", "HDFCLIFE", "BPCL", "TRENT", "BRITANNIA", "CIPLA",
    "TATACONSUM", "BAJAJ-AUTO", "HINDALCO", "DIVISLAB", "EICHERMOT", "APOLLOHOSP",
    "SHRIRAMFIN", "BEL"
]

# Comprehensive liquid F&O underlying universe
NSE_FO_SEED = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT",
    "ADANIPORTS", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BALKRISIND", "BALRAMCHIN",
    "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", "BHARATFORG",
    "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSOFT",
    "CANBK", "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND",
    "DABUR", "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF",
    "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL",
    "GLENMARK", "GMRAIRPORT", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES",
    "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDPETRO", "HINDUNILVR", "ICICIBANK",
    "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL",
    "INDIAMART", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", "JIOFIN", "JKCEMENT",
    "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LTF", "LALPATHLAB", "LAURUSLABS",
    "LICHSGFIN", "LICI", "LT", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM",
    "MARICO", "MARUTI", "UNITDSPR", "MCX", "METROPOLIS", "MFSL", "MGL",
    "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", "NAUKRI",
    "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS", "ONGC",
    "PAGEIND", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND",
    "PNB", "POLYCAB", "POWERGRID", "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD",
    "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SRF", "SUNPHARMA", "SUNTV", "SYNGENE", "TATACHEM", "TATACOMM",
    "TATACONSUM", "TATAELXSI", "TATAPOWER", "TATASTEEL", "TATATECH", "TECHM",
    "TITAN", "TMPV", "TORNTPHARM", "TORNTPOWER", "TRENT", "TVSMOTOR", "UBL",
    "ULTRACEMCO", "UNIONBANK", "UPL", "VEDL", "VOLTAS", "WIPRO"
]

# Major liquid MCX commodity symbols
MCX_COMMODITIES_SEED = [
    "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER", "ZINC", "ALUMINIUM"
]

# Major Indian Market & Sectoral Indices
INDICES_SEED = [
    "NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY", "MIDCPNIFTY", "NIFTYIT", "INDIAVIX"
]

# Mapping MCX commodity symbols to Yahoo Finance futures / ETF tickers for fallback
MCX_TO_YFINANCE = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDEOIL": "CL=F",
    "NATURALGAS": "NG=F",
    "COPPER": "HG=F",
    "ZINC": "ZNC=F",
    "ALUMINIUM": "ALI=F",
}

YFINANCE_TO_MCX = {v: k for k, v in MCX_TO_YFINANCE.items()}

# Mapping Indian Indices to Yahoo Finance symbols
INDEX_TO_YFINANCE = {
    "NIFTY": "^NSEI",
    "NIFTY50": "^NSEI",
    "NIFTY_50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "BANK_NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "FIN_NIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSEMDCP50",
    "MIDCAP_NIFTY": "^NSEMDCP50",
    "NIFTYIT": "^CNXIT",
    "NIFTY_IT": "^CNXIT",
    "INDIAVIX": "^INDIAVIX",
    "INDIA_VIX": "^INDIAVIX",
}

YFINANCE_TO_INDEX = {
    "^NSEI": "NIFTY",
    "^NSEBANK": "BANKNIFTY",
    "^BSESN": "SENSEX",
    "NIFTY_FIN_SERVICE.NS": "FINNIFTY",
    "^NSEMDCP50": "MIDCPNIFTY",
    "^CNXIT": "NIFTYIT",
    "^INDIAVIX": "INDIAVIX",
}


class UniverseManager:
    """Manages stock universe definitions and handles exchange ticker conversions."""

    @staticmethod
    def get_tickers(universe_name: str = "NIFTY_50") -> List[str]:
        """
        Loads symbols for a chosen universe:
        Options: 'NIFTY_50', 'NIFTY_500', 'NSE_FO', 'MCX_COMMODITIES', 'INDICES', or custom path/comma-separated string.
        """
        raw_name = universe_name.strip().upper()
        u_norm = raw_name.replace(" ", "_").replace("-", "_")

        if u_norm in ["NIFTY_50", "NIFTY50"]:
            return UniverseManager._load_nifty_50()
        elif u_norm in ["NIFTY_500", "NIFTY500"]:
            return UniverseManager._load_nifty_500()
        elif u_norm in ["NSE_FO", "NIFTY_FO", "FNO", "FO"]:
            return UniverseManager._load_fno()
        elif u_norm in ["MCX", "MCX_COMMODITIES", "COMMODITIES", "COMMODITY"]:
            return UniverseManager._load_mcx_commodities()
        elif any(k in u_norm for k in ["INDICES", "INDEX", "MARKET_INDICES", "SECTORAL_INDICES"]):
            return UniverseManager._load_indices()
        else:
            # Check if file exists in universes directory
            file_path = UNIVERSES_DIR / f"{universe_name.lower()}.txt"
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
            
            # Treat as comma-separated tickers if passed directly
            if "," in universe_name or " " in universe_name:
                return [s.strip().upper() for s in universe_name.replace(",", " ").split() if s.strip()]

            return UniverseManager._load_nifty_50()

    @staticmethod
    def _load_nifty_50() -> List[str]:
        file_path = UNIVERSES_DIR / "nifty_50.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
                if len(symbols) >= 45:
                    return symbols

        # Attempt downloading live official Nifty 50 CSV from NSE
        csv_url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            resp = requests.get(csv_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(pd.io.common.StringIO(resp.text))
                if "Symbol" in df.columns:
                    symbols = df["Symbol"].dropna().str.strip().str.upper().tolist()
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(symbols))
                    return symbols
        except Exception:
            pass

        # Write reliable seed
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(NIFTY_50_SEED))
        return NIFTY_50_SEED

    @staticmethod
    def _load_nifty_500() -> List[str]:
        file_path = UNIVERSES_DIR / "nifty_500.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
                if len(symbols) >= 100:
                    return symbols

        # Attempt downloading official Nifty 500 CSV from NSE
        csv_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            resp = requests.get(csv_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(pd.io.common.StringIO(resp.text))
                if "Symbol" in df.columns:
                    symbols = df["Symbol"].dropna().str.strip().str.upper().tolist()
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(symbols))
                    return symbols
        except Exception:
            pass

        # Fallback to Nifty 50
        return UniverseManager._load_nifty_50()

    @staticmethod
    def _load_fno() -> List[str]:
        """Loads liquid F&O underlying stocks."""
        file_path = UNIVERSES_DIR / "nse_fo.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
                if len(symbols) >= 50:
                    return symbols

        # Save the full F&O seed list
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(NSE_FO_SEED))
        return NSE_FO_SEED

    @staticmethod
    def _load_indices() -> List[str]:
        """Loads major Indian market & sectoral index symbols."""
        file_path = UNIVERSES_DIR / "indices.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
                if len(symbols) >= 3:
                    return symbols

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(INDICES_SEED))
        return INDICES_SEED

    @staticmethod
    def _load_mcx_commodities() -> List[str]:
        """Loads liquid MCX commodity symbols."""
        file_path = UNIVERSES_DIR / "mcx_commodities.txt"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]
                if len(symbols) >= 3:
                    return symbols

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(MCX_COMMODITIES_SEED))
        return MCX_COMMODITIES_SEED

    @staticmethod
    def is_commodity(symbol: str) -> bool:
        """Checks if a symbol is an MCX commodity or mapped commodity future."""
        clean = UniverseManager.to_clean_symbol(symbol)
        return clean in MCX_COMMODITIES_SEED or symbol.upper() in MCX_TO_YFINANCE or symbol.upper() in YFINANCE_TO_MCX

    @staticmethod
    def is_index(symbol: str) -> bool:
        """Checks if a symbol is an Indian market index (e.g. NIFTY, BANKNIFTY, SENSEX)."""
        clean = UniverseManager.to_clean_symbol(symbol)
        return clean in INDICES_SEED or symbol.upper() in INDEX_TO_YFINANCE or symbol.startswith("^")

    @staticmethod
    def is_fno(symbol: str) -> bool:
        """Checks if a symbol is in the NSE F&O (Futures & Options) underlying universe."""
        clean = UniverseManager.to_clean_symbol(symbol)
        return clean in NSE_FO_SEED or clean in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

    @staticmethod
    def get_exchange(symbol: str) -> str:
        """Returns the primary exchange for the asset ('MCX', 'BSE', or 'NSE')."""
        clean = UniverseManager.to_clean_symbol(symbol)
        if UniverseManager.is_commodity(symbol):
            return "MCX"
        if clean in ["SENSEX", "BSESN"] or symbol.upper().endswith(".BO"):
            return "BSE"
        return "NSE"

    @staticmethod
    def to_yfinance_symbol(symbol: str, exchange: str = "NSE") -> str:
        """
        Converts symbol to Yahoo Finance ticker format.
        Maps Indian indices (e.g. NIFTY -> ^NSEI, BANKNIFTY -> ^NSEBANK, SENSEX -> ^BSESN),
        MCX commodities (e.g. GOLD -> GC=F, CRUDEOIL -> CL=F),
        and Indian equities (e.g. RELIANCE -> RELIANCE.NS).
        """
        clean = UniverseManager.to_clean_symbol(symbol)
        if clean in INDEX_TO_YFINANCE:
            return INDEX_TO_YFINANCE[clean]
        if clean in MCX_TO_YFINANCE:
            return MCX_TO_YFINANCE[clean]

        if symbol.startswith("^") or symbol.endswith(".NS") or symbol.endswith(".BO") or "=" in symbol:
            return symbol

        suffix = ".BO" if exchange.upper() == "BSE" or clean == "SENSEX" else ".NS"
        return f"{clean}{suffix}"

    @staticmethod
    def to_clean_symbol(symbol: str) -> str:
        """Strips exchange suffixes (.NS, .BO, =F, ^) and returns clean base code."""
        s = symbol.upper()
        if s in YFINANCE_TO_INDEX:
            return YFINANCE_TO_INDEX[s]
        if s in YFINANCE_TO_MCX:
            return YFINANCE_TO_MCX[s]
        clean = s.replace(".NS", "").replace(".BO", "").replace("^", "")
        return clean

    @staticmethod
    def get_commodity_unit(symbol: str) -> str:
        """Returns standard official Multi Commodity Exchange of India (MCX) contract quotation unit."""
        clean = UniverseManager.to_clean_symbol(symbol)
        units = {
            "GOLD": "₹ / 10g",
            "SILVER": "₹ / kg",
            "CRUDEOIL": "₹ / bbl",
            "NATURALGAS": "₹ / mmBtu",
            "COPPER": "₹ / kg",
            "ZINC": "₹ / kg",
            "ALUMINIUM": "₹ / kg",
        }
        return units.get(clean, "MCX")

    @staticmethod
    def get_usdinr_rate() -> float:
        """
        Retrieves real-time USD/INR exchange rate with 5-minute memory caching.
        Falls back to resilient default (95.5) if offline or network throttled.
        """
        now = time.time()
        if now - _USDINR_CACHE["timestamp"] < 300:
            return _USDINR_CACHE["rate"]
        try:
            tk = yf.Ticker("USDINR=X")
            fi = tk.fast_info
            p = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
            if p and p > 50:
                _USDINR_CACHE["rate"] = float(p)
                _USDINR_CACHE["timestamp"] = now
                return float(p)
        except Exception:
            pass
        return _USDINR_CACHE["rate"]

    @staticmethod
    def get_mcx_conversion_multiplier(symbol: str, raw_price: float, usdinr: Optional[float] = None) -> float:
        """
        Calculates exact mathematical multiplier to convert unadjusted international
        futures (COMEX/NYMEX/LME) into authoritative Multi Commodity Exchange of India (MCX)
        quotation units and INR (₹).

        Contract specifications:
          - GOLD (MCX: ₹ / 10g): COMEX GC=F ($/troy oz) * (10 / 31.1034768) * USDINR * 1.10 (customs duty + AIDC landed parity)
          - SILVER (MCX: ₹ / kg): COMEX SI=F ($/troy oz) * (1000 / 31.1034768) * USDINR * 1.15 (import tariff landed parity)
          - CRUDEOIL (MCX: ₹ / bbl): NYMEX CL=F ($/bbl) * USDINR
          - NATURALGAS (MCX: ₹ / mmBtu): NYMEX NG=F ($/mmBtu) * USDINR
          - COPPER (MCX: ₹ / kg): COMEX HG=F ($/lb) * 2.20462262 * USDINR
          - ZINC (MCX: ₹ / kg): LME ZNC=F ($/MT) / 1000 * USDINR (or $/lb * 2.20462 * USDINR)
          - ALUMINIUM (MCX: ₹ / kg): LME ALI=F ($/MT) / 1000 * USDINR (or $/lb * 2.20462 * USDINR)
        """
        clean = UniverseManager.to_clean_symbol(symbol)
        if not UniverseManager.is_commodity(clean) or raw_price is None or raw_price <= 0:
            return 1.0

        if usdinr is None:
            usdinr = UniverseManager.get_usdinr_rate()

        if clean == "GOLD":
            # If price > 20000, it's already in MCX INR per 10g
            if raw_price > 20000:
                return 1.0
            # Standard COMEX GC=F is $/Troy Oz
            return (10.0 / 31.1034768) * usdinr * 1.10

        elif clean == "SILVER":
            # If price > 20000, it's already in MCX INR per kg
            if raw_price > 20000:
                return 1.0
            # COMEX SI=F is $/Troy Oz
            if raw_price <= 500:
                return (1000.0 / 31.1034768) * usdinr * 1.15
            return 1.0

        elif clean == "CRUDEOIL":
            # If price > 1000, it's already in MCX INR per bbl
            if raw_price > 1000:
                return 1.0
            # NYMEX CL=F is $/barrel
            return usdinr

        elif clean == "NATURALGAS":
            # If price > 50, it's already in MCX INR per mmBtu
            if raw_price > 50:
                return 1.0
            # NYMEX NG=F is $/mmBtu
            return usdinr

        elif clean == "COPPER":
            # If price > 100, it's already in MCX INR per kg
            if raw_price > 100:
                return 1.0
            # COMEX HG=F is $/lb (1 lb = 0.45359237 kg -> 1 kg = 2.20462262 lbs)
            return 2.20462262 * usdinr

        elif clean == "ZINC":
            # If 50 < price < 500, it's already in MCX INR per kg
            if 50 < raw_price < 500:
                return 1.0
            if raw_price >= 500:
                # LME USD per metric ton -> kg
                return (1.0 / 1000.0) * usdinr
            # Quoted in $/lb
            return 2.20462262 * usdinr if raw_price < 10 else usdinr

        elif clean == "ALUMINIUM":
            # If 50 < price < 500, it's already in MCX INR per kg
            if 50 < raw_price < 500:
                return 1.0
            if raw_price >= 500:
                # LME USD per metric ton -> kg
                return (1.0 / 1000.0) * usdinr
            # Quoted in $/lb
            return 2.20462262 * usdinr if raw_price < 10 else usdinr

        return 1.0

    @staticmethod
    def convert_ohlcv_to_mcx(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts OHLCV candlestick data of international commodity futures to Indian MCX INR.
        Preserves 100% of candle shapes, volatility, technical indicator patterns, and wicks.
        """
        clean = UniverseManager.to_clean_symbol(symbol)
        if not UniverseManager.is_commodity(clean) or df is None or df.empty:
            return df

        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
        if not price_cols:
            return df

        df = df.copy()
        valid_closes = df['close'].dropna()
        if valid_closes.empty:
            return df

        last_close = float(valid_closes.iloc[-1])
        multiplier = UniverseManager.get_mcx_conversion_multiplier(clean, last_close)
        if abs(multiplier - 1.0) > 1e-4:
            for col in price_cols:
                df[col] = (df[col] * multiplier).round(2)
        return df


def get_usdinr_rate() -> float:
    """Module-level function returning current USD/INR exchange rate."""
    return UniverseManager.get_usdinr_rate()

