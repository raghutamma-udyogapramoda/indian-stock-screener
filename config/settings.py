import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
UNIVERSES_DIR = CONFIG_DIR / "universes"
CACHE_DIR = BASE_DIR / "data" / "cache"
STATIC_BREEZE_DIR = BASE_DIR / "data" / "static_breeze"

# Ensure runtime directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_BREEZE_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables (.env)
load_dotenv(BASE_DIR / ".env")

# ICICI Direct Breeze API Credentials
BREEZE_API_KEY = os.getenv("BREEZE_API_KEY", "")
BREEZE_SECRET_KEY = os.getenv("BREEZE_SECRET_KEY", "")
BREEZE_SESSION_TOKEN = os.getenv("BREEZE_SESSION_TOKEN", "")

# Gemini AI Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Screening Thresholds
DEFAULTS = {
    "MIN_PRICE": 10.0,            # Filter out illiquid penny stocks
    "MIN_AVG_VOLUME": 100000,      # Minimum 20-day average volume
    "RVOL_THRESHOLD": 1.8,         # 1.8x 20-day average volume for surges
    "CLV_BTST_THRESHOLD": 0.82,    # Closed in top 18% of day's range
    "BB_SQUEEZE_LOOKBACK": 40,     # Squeeze lookback in trading sessions
    "MAX_CANDIDATES_FOR_AI": 8,    # Max candidates per category passed to LLM
}
