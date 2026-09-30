"""
Vectorized Technical Indicators implemented in pure Pandas & NumPy.
Eliminates external C-library dependency (TA-Lib) ensuring 100% portability.
"""

import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, length: int = 20) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length, min_periods=length).mean()


def compute_ema(series: pd.Series, length: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()


def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's Smoothed Relative Strength Index (RSI).
    Standard 14-period RSI used in institutional momentum trading.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # First value is simple average
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_bollinger_bands(close: pd.Series, length: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands: Middle, Upper, Lower, and Bandwidth (BBW).
    BBW = (Upper - Lower) / Middle
    Low BBW indicates high volatility compression (squeeze).
    """
    middle = compute_sma(close, length)
    std = close.rolling(window=length, min_periods=length).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = (upper - lower) / (middle + 1e-10)

    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth
    }, index=close.index)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average True Range (ATR) based on Wilder's Smoothing.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return atr


def compute_clv(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Close Location Value (CLV) / Normalized Close:
    CLV = (Close - Low) / (High - Low)
    Values >= 0.80 indicate strong finish near the day's high (buying pressure).
    Values <= 0.20 indicate heavy selling into the close.
    """
    rng = high - low
    clv = np.where(rng > 1e-6, (close - low) / rng, 0.5)
    return pd.Series(clv, index=close.index)


def compute_rvol(volume: pd.Series, length: int = 20) -> pd.Series:
    """
    Relative Volume (RVol) = Current Volume / 20-period SMA Volume.
    RVol > 1.8 indicates heavy institutional accumulation or distribution.
    """
    vol_sma = compute_sma(volume, length)
    return volume / (vol_sma + 1e-6)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP) for intraday data.
    df must contain: 'high', 'low', 'close', 'volume'
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_vol_price = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    return cum_vol_price / (cum_vol + 1e-6)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    Level-II formation: macd_line > macd_signal and macd_line > 0 (strong bullish momentum).
    """
    fast_ema = compute_ema(close, fast)
    slow_ema = compute_ema(close, slow)
    macd_line = fast_ema - slow_ema
    macd_signal = compute_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist
    }, index=close.index)


def compute_aroon(high: pd.Series, low: pd.Series, length: int = 25) -> pd.DataFrame:
    """
    Aroon Indicator & Aroon Oscillator (Tushar Chande).
    Measures trend strength and anticipation of breakout/reversals.
    Aroon Up = ((length - periods since highest high) / length) * 100
    Aroon Down = ((length - periods since lowest low) / length) * 100
    Aroon Oscillator = Aroon Up - Aroon Down. Values > +50 indicate strong bullish trend.
    """
    def _periods_since_max(x):
        return (len(x) - 1) - int(np.argmax(x))

    def _periods_since_min(x):
        return (len(x) - 1) - int(np.argmin(x))

    high_idx = high.rolling(window=length, min_periods=length).apply(_periods_since_max, raw=True)
    low_idx = low.rolling(window=length, min_periods=length).apply(_periods_since_min, raw=True)

    aroon_up = ((length - high_idx) / float(length)) * 100.0
    aroon_down = ((length - low_idx) / float(length)) * 100.0
    aroon_osc = aroon_up - aroon_down

    return pd.DataFrame({
        "aroon_up": aroon_up,
        "aroon_down": aroon_down,
        "aroon_osc": aroon_osc
    }, index=high.index)


def enrich_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches an OHLCV DataFrame with all standard metrics in one vectorized pass.
    Expected columns: 'open', 'high', 'low', 'close', 'volume' (lowercase)

    These indicators directly power the quantitative screening engines:
      - Trend (EMA 9, 20, 50, SMA 200): Used for Stage-2 uptrend (Breakout/Swing)
        and Stage-4 downtrend (Breakdown/Short).
      - Momentum (RSI 14, MACD Level-II, Aroon Oscillator):
        * RSI > 55 in BTST/Breakout; 38-64 in Swing pullbacks.
        * MACD Level-II: MACD > Signal > 0 (bullish continuation).
        * Aroon Oscillator > +50: Strong institutional trend dominance.
      - Volatility (Bollinger Bands & Bandwidth BBW): Low BBW indicates volatility
        squeezes (imminent breakout explosion).
      - Risk Sizing (ATR 14): Used by AI Advisor to size stop-loss and targets dynamically.
      - Volume (RVol 20): Detects institutional participation (>1.3x 20-day volume).
      - Close Location Value (CLV): Measures intra-session buying pressure (CLV >= 0.70)
        vs institutional dumping (CLV <= 0.35).
    """
    if df is None or len(df) < 20:
        return df

    # Trend EMAs & SMAs: Stage analysis and dynamic support/resistance
    df['ema_9'] = compute_ema(df['close'], 9)
    df['ema_20'] = compute_ema(df['close'], 20)
    df['ema_50'] = compute_ema(df['close'], 50)
    df['sma_200'] = compute_sma(df['close'], 200)

    # Momentum: Wilder's smoothed RSI (14 periods)
    df['rsi_14'] = compute_rsi(df['close'], 14)

    # MACD (12, 26, 9)
    macd_df = compute_macd(df['close'], 12, 26, 9)
    df['macd_line'] = macd_df['macd_line']
    df['macd_signal'] = macd_df['macd_signal']
    df['macd_hist'] = macd_df['macd_hist']

    # Aroon Indicator (25 periods)
    aroon_df = compute_aroon(df['high'], df['low'], 25)
    df['aroon_up'] = aroon_df['aroon_up']
    df['aroon_down'] = aroon_df['aroon_down']
    df['aroon_osc'] = aroon_df['aroon_osc']

    # Volatility: 20 SMA with 2 standard deviations + Bandwidth
    bb = compute_bollinger_bands(df['close'], 20, 2.0)
    df['bb_upper'] = bb['bb_upper']
    df['bb_middle'] = bb['bb_middle']
    df['bb_lower'] = bb['bb_lower']
    df['bb_bandwidth'] = bb['bb_bandwidth']
    df['atr_14'] = compute_atr(df['high'], df['low'], df['close'], 14)

    # Volume & Price Action: 9-day and 20-day Volume SMAs, RVols, and Normalized Close
    df['volume_sma_9'] = compute_sma(df['volume'], 9)
    df['rvol_9'] = compute_rvol(df['volume'], 9)
    df['volume_sma_20'] = compute_sma(df['volume'], 20)
    df['rvol_20'] = compute_rvol(df['volume'], 20)
    df['clv'] = compute_clv(df['high'], df['low'], df['close'])

    return df
