"""
Relative Strength, Sector Tailwind, Panic-Day Resilience, and Retracement Analytics.
Provides quantitative modules for:
  1. Retracement Check: 50% Candle Body Defense Rule (Invalidation & Dip-Buy Zones).
  2. Sector Tailwind: Alignment of stock trend with its sector benchmark (NIFTYIT, BANKNIFTY, MIDCPNIFTY, NIFTY).
  3. Panic-Day Behavior: Institutional accumulation & relative strength during market selloffs.
  4. Mansfield Relative Strength (MRS): 50-period outperformance vs NIFTY 50.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


# Comprehensive mapping of liquid Indian Equities to their sector & benchmark index
STOCK_SECTOR_MAP = {
    # IT / Technology -> NIFTYIT
    "TCS": ("IT & Tech", "NIFTYIT"),
    "INFY": ("IT & Tech", "NIFTYIT"),
    "WIPRO": ("IT & Tech", "NIFTYIT"),
    "HCLTECH": ("IT & Tech", "NIFTYIT"),
    "TECHM": ("IT & Tech", "NIFTYIT"),
    "LTIM": ("IT & Tech", "NIFTYIT"),
    "PERSISTENT": ("IT & Tech", "NIFTYIT"),
    "COFORGE": ("IT & Tech", "NIFTYIT"),
    "MPHASIS": ("IT & Tech", "NIFTYIT"),
    "LTTS": ("IT & Tech", "NIFTYIT"),
    "TATAELXSI": ("IT & Tech", "NIFTYIT"),
    "BSOFT": ("IT & Tech", "NIFTYIT"),
    "OFSS": ("IT & Tech", "NIFTYIT"),
    "NAUKRI": ("IT & Tech", "NIFTYIT"),

    # Banking & Financial Services -> BANKNIFTY / FINNIFTY
    "HDFCBANK": ("Banking & Financials", "BANKNIFTY"),
    "ICICIBANK": ("Banking & Financials", "BANKNIFTY"),
    "SBIN": ("Banking & Financials", "BANKNIFTY"),
    "KOTAKBANK": ("Banking & Financials", "BANKNIFTY"),
    "AXISBANK": ("Banking & Financials", "BANKNIFTY"),
    "INDUSINDBK": ("Banking & Financials", "BANKNIFTY"),
    "BANKBARODA": ("Banking & Financials", "BANKNIFTY"),
    "PNB": ("Banking & Financials", "BANKNIFTY"),
    "CANBK": ("Banking & Financials", "BANKNIFTY"),
    "FEDERALBNK": ("Banking & Financials", "BANKNIFTY"),
    "IDFCFIRSTB": ("Banking & Financials", "BANKNIFTY"),
    "AUBANK": ("Banking & Financials", "BANKNIFTY"),
    "BANDHANBNK": ("Banking & Financials", "BANKNIFTY"),
    "RBLBANK": ("Banking & Financials", "BANKNIFTY"),
    "UNIONBANK": ("Banking & Financials", "BANKNIFTY"),
    "BAJFINANCE": ("Banking & Financials", "BANKNIFTY"),
    "BAJAJFINSV": ("Banking & Financials", "BANKNIFTY"),
    "CHOLAFIN": ("Banking & Financials", "BANKNIFTY"),
    "MUTHOOTFIN": ("Banking & Financials", "BANKNIFTY"),
    "SHRIRAMFIN": ("Banking & Financials", "BANKNIFTY"),
    "PFC": ("Banking & Financials", "BANKNIFTY"),
    "RECLTD": ("Banking & Financials", "BANKNIFTY"),
    "LICI": ("Banking & Financials", "BANKNIFTY"),
    "SBICARD": ("Banking & Financials", "BANKNIFTY"),
    "SBILIFE": ("Banking & Financials", "BANKNIFTY"),
    "HDFCLIFE": ("Banking & Financials", "BANKNIFTY"),
    "ICICIPRULI": ("Banking & Financials", "BANKNIFTY"),
    "ICICIGI": ("Banking & Financials", "BANKNIFTY"),
    "HDFCAMC": ("Banking & Financials", "BANKNIFTY"),
    "ABCAPITAL": ("Banking & Financials", "BANKNIFTY"),
    "CANFINHOME": ("Banking & Financials", "BANKNIFTY"),
    "LICHSGFIN": ("Banking & Financials", "BANKNIFTY"),
    "M&MFIN": ("Banking & Financials", "BANKNIFTY"),
    "MANAPPURAM": ("Banking & Financials", "BANKNIFTY"),
    "JIOFIN": ("Banking & Financials", "BANKNIFTY"),

    # Healthcare & Pharma -> NIFTY (Healthcare tailwind)
    "SUNPHARMA": ("Pharma & Healthcare", "NIFTY"),
    "CIPLA": ("Pharma & Healthcare", "NIFTY"),
    "DRREDDY": ("Pharma & Healthcare", "NIFTY"),
    "DIVISLAB": ("Pharma & Healthcare", "NIFTY"),
    "LUPIN": ("Pharma & Healthcare", "NIFTY"),
    "AUROPHARMA": ("Pharma & Healthcare", "NIFTY"),
    "TORNTPHARM": ("Pharma & Healthcare", "NIFTY"),
    "ALKEM": ("Pharma & Healthcare", "NIFTY"),
    "APOLLOHOSP": ("Pharma & Healthcare", "NIFTY"),
    "LALPATHLAB": ("Pharma & Healthcare", "NIFTY"),
    "METROPOLIS": ("Pharma & Healthcare", "NIFTY"),
    "LAURUSLABS": ("Pharma & Healthcare", "NIFTY"),
    "BIOCON": ("Pharma & Healthcare", "NIFTY"),
    "GLENMARK": ("Pharma & Healthcare", "NIFTY"),
    "GRANULES": ("Pharma & Healthcare", "NIFTY"),
    "IPCALAB": ("Pharma & Healthcare", "NIFTY"),
    "ABBOTINDIA": ("Pharma & Healthcare", "NIFTY"),
    "SYNGENE": ("Pharma & Healthcare", "NIFTY"),

    # Automobile & Mobility -> NIFTY
    "MARUTI": ("Automobile & Auto Parts", "NIFTY"),
    "M&M": ("Automobile & Auto Parts", "NIFTY"),
    "BAJAJ-AUTO": ("Automobile & Auto Parts", "NIFTY"),
    "TATAMOTORS": ("Automobile & Auto Parts", "NIFTY"),
    "TMPV": ("Automobile & Auto Parts", "NIFTY"),
    "EICHERMOT": ("Automobile & Auto Parts", "NIFTY"),
    "HEROMOTOCO": ("Automobile & Auto Parts", "NIFTY"),
    "TVSMOTOR": ("Automobile & Auto Parts", "NIFTY"),
    "ASHOKLEY": ("Automobile & Auto Parts", "NIFTY"),
    "BALKRISIND": ("Automobile & Auto Parts", "NIFTY"),
    "MRF": ("Automobile & Auto Parts", "NIFTY"),
    "BHARATFORG": ("Automobile & Auto Parts", "NIFTY"),
    "BOSCHLTD": ("Automobile & Auto Parts", "NIFTY"),
    "APOLLOTYRE": ("Automobile & Auto Parts", "NIFTY"),
    "EXIDEIND": ("Automobile & Auto Parts", "NIFTY"),
    "MOTHERSON": ("Automobile & Auto Parts", "NIFTY"),

    # Metal & Mining -> NIFTY
    "TATASTEEL": ("Metals & Mining", "NIFTY"),
    "JSWSTEEL": ("Metals & Mining", "NIFTY"),
    "HINDALCO": ("Metals & Mining", "NIFTY"),
    "VEDL": ("Metals & Mining", "NIFTY"),
    "JINDALSTEL": ("Metals & Mining", "NIFTY"),
    "COALINDIA": ("Metals & Mining", "NIFTY"),
    "NMDC": ("Metals & Mining", "NIFTY"),
    "SAIL": ("Metals & Mining", "NIFTY"),
    "NATIONALUM": ("Metals & Mining", "NIFTY"),

    # Energy, Oil & Gas -> NIFTY
    "RELIANCE": ("Energy, Oil & Gas", "NIFTY"),
    "ONGC": ("Energy, Oil & Gas", "NIFTY"),
    "BPCL": ("Energy, Oil & Gas", "NIFTY"),
    "IOC": ("Energy, Oil & Gas", "NIFTY"),
    "HINDPETRO": ("Energy, Oil & Gas", "NIFTY"),
    "GAIL": ("Energy, Oil & Gas", "NIFTY"),
    "PETRONET": ("Energy, Oil & Gas", "NIFTY"),
    "MGL": ("Energy, Oil & Gas", "NIFTY"),
    "IGL": ("Energy, Oil & Gas", "NIFTY"),
    "NTPC": ("Power & Energy", "NIFTY"),
    "POWERGRID": ("Power & Energy", "NIFTY"),
    "TATAPOWER": ("Power & Energy", "NIFTY"),
    "TORNTPOWER": ("Power & Energy", "NIFTY"),

    # FMCG & Consumer -> NIFTY
    "ITC": ("FMCG & Consumption", "NIFTY"),
    "HINDUNILVR": ("FMCG & Consumption", "NIFTY"),
    "NESTLEIND": ("FMCG & Consumption", "NIFTY"),
    "BRITANNIA": ("FMCG & Consumption", "NIFTY"),
    "TATACONSUM": ("FMCG & Consumption", "NIFTY"),
    "DABUR": ("FMCG & Consumption", "NIFTY"),
    "MARICO": ("FMCG & Consumption", "NIFTY"),
    "COLPAL": ("FMCG & Consumption", "NIFTY"),
    "GODREJCP": ("FMCG & Consumption", "NIFTY"),
    "UNITDSPR": ("FMCG & Consumption", "NIFTY"),
    "UBL": ("FMCG & Consumption", "NIFTY"),
    "TITAN": ("Consumer Discretionary", "NIFTY"),
    "TRENT": ("Consumer Discretionary", "NIFTY"),
    "PAGEIND": ("Consumer Discretionary", "NIFTY"),
    "BATAINDIA": ("Consumer Discretionary", "NIFTY"),
    "ASIANPAINT": ("Consumer Discretionary", "NIFTY"),
    "BERGEPAINT": ("Consumer Discretionary", "NIFTY"),
    "PIDILITIND": ("Consumer Discretionary", "NIFTY"),
    "HAVELLS": ("Consumer Discretionary", "NIFTY"),
    "VOLTAS": ("Consumer Discretionary", "NIFTY"),
    "DIXON": ("Consumer Discretionary", "NIFTY"),
    "POLYCAB": ("Consumer Discretionary", "NIFTY"),

    # Infrastructure, Capital Goods, Realty -> NIFTY
    "LT": ("Infrastructure & Capital Goods", "NIFTY"),
    "SIEMENS": ("Infrastructure & Capital Goods", "NIFTY"),
    "ABB": ("Infrastructure & Capital Goods", "NIFTY"),
    "BHEL": ("Infrastructure & Capital Goods", "NIFTY"),
    "CUMMINSIND": ("Infrastructure & Capital Goods", "NIFTY"),
    "HAL": ("Defense & Aerospace", "NIFTY"),
    "BEL": ("Defense & Aerospace", "NIFTY"),
    "DLF": ("Real Estate", "NIFTY"),
    "GODREJPROP": ("Real Estate", "NIFTY"),
    "OBEROIRLTY": ("Real Estate", "NIFTY"),

    # Telecom & Chemicals
    "BHARTIARTL": ("Telecom", "NIFTY"),
    "INDUSTOWER": ("Telecom", "NIFTY"),
    "IDEA": ("Telecom", "NIFTY"),
    "SRF": ("Chemicals", "NIFTY"),
    "DEEPAKNTR": ("Chemicals", "NIFTY"),
    "TATACHEM": ("Chemicals", "NIFTY"),
    "PIIND": ("Chemicals", "NIFTY"),
    "AARTIIND": ("Chemicals", "NIFTY"),
    "ATUL": ("Chemicals", "NIFTY"),
    "NAVINFLUOR": ("Chemicals", "NIFTY"),
    "UPL": ("Chemicals", "NIFTY"),
}


class MarketIntelAnalyzer:
    """
    Evaluates:
      1. Retracement Check: 50% Breakout Candle Body Defense Rule.
      2. Sector Tailwind: Trend alignment of parent index.
      3. Panic-Day Behavior: Relative strength during broad market drawdowns.
      4. Mansfield Relative Strength: Medium-term alpha vs NIFTY 50.
    """

    @staticmethod
    def get_stock_sector(symbol: str) -> Tuple[str, str]:
        """Returns (sector_name, benchmark_symbol). Defaults to ('Broad Market', 'NIFTY')."""
        clean_sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
        return STOCK_SECTOR_MAP.get(clean_sym, ("Broad Market", "NIFTY"))

    @classmethod
    def get_sector(cls, symbol: str) -> str:
        """Returns just the sector name for a symbol."""
        sector, _ = cls.get_stock_sector(symbol)
        return sector

    @staticmethod
    def evaluate_retracement(df: pd.DataFrame, is_bullish: bool = True) -> dict:
        """
        Calculates the 50% Candle Body Defense Level and assesses retracement health.
        
        Institutional Principle:
          - Breakout Candle Body = abs(close - open).
          - Retracement Defense Level (50% midpoint) = open + 0.50 * (close - open) for bullish.
          - If price stays above this 50% body level on intraday pullbacks or Day-2 retests,
            the setup exhibits institutional absorption and pristine support.
          - If price closes or drops deeply below 50% of the body, the breakout has lost
            conviction and risks bull-trap invalidation.
        """
        if df is None or len(df) < 2:
            return {
                "retracement_healthy": True,
                "retest_support_50pct": 0.0,
                "candle_body_pts": 0.0,
                "retracement_ratio_pct": 0.0,
                "defense_desc": "Insufficient candle history",
                "invalidation_sl": 0.0
            }

        last = df.iloc[-1]
        prev = df.iloc[-2]

        last_o = float(last['open'])
        last_c = float(last['close'])
        last_h = float(last['high'])
        last_l = float(last['low'])

        prev_o = float(prev['open'])
        prev_c = float(prev['close'])

        if is_bullish:
            # Check if today is the primary impulse/breakout candle or a retest day
            candle_body = max(last_c - last_o, 0.01)
            is_green_today = last_c >= last_o
            
            # 50% midpoint of today's candle body
            today_midpoint = round(last_o + (0.50 * candle_body), 2)
            
            # Check previous candle if yesterday was the larger impulse bar
            prev_body = max(prev_c - prev_o, 0.01)
            prev_midpoint = round(prev_o + (0.50 * prev_body), 2)

            # Determine reference breakout bar:
            # If today is expanding (>1.5% green), today's body is the reference.
            # If today is a tight consolidation/pullback day after yesterday's big bar, yesterday is reference.
            if is_green_today and (last_c >= last_o * 1.012 or prev_c <= prev_o):
                retest_level = today_midpoint
                upper_wick = max(last_h - last_c, 0.0)
                retracement_ratio = (upper_wick / (candle_body + 1e-6)) * 100.0
                
                # Healthy if close finished in upper half of body and didn't suffer >45% upper wick rejection
                is_healthy = (last_c >= today_midpoint) and (retracement_ratio <= 48.0)
                invalidation_sl = round(today_midpoint * 0.995, 1)

                if is_healthy:
                    defense_desc = (
                        f"Retracement Intact: Defending >50% breakout candle body (Support: ₹{today_midpoint:,.1f}). "
                        f"Dip buy zone: ₹{today_midpoint:,.1f} – ₹{last_c:,.1f} (SL: ₹{invalidation_sl:,.1f})"
                    )
                else:
                    defense_desc = (
                        f"Weak Retracement: Rejection from high (Retraced {retracement_ratio:.0f}% into candle body). "
                        f"Caution: Needs closing hold above ₹{today_midpoint:,.1f}"
                    )
            else:
                # Retest of yesterday's breakout bar
                retest_level = prev_midpoint
                invalidation_sl = round(prev_midpoint * 0.995, 1)
                is_healthy = (last_l >= prev_midpoint * 0.997) and (last_c >= prev_midpoint)
                retracement_ratio = round(((prev_c - last_c) / (prev_body + 1e-6)) * 100.0, 1)

                if is_healthy:
                    defense_desc = (
                        f"Pristine Retest: Low ₹{last_l:,.1f} defended 50% midpoint of prior breakout body (₹{prev_midpoint:,.1f}). "
                        f"High-probability dip absorption zone."
                    )
                else:
                    defense_desc = (
                        f"Deep Pullback: Penetrated below 50% of prior breakout body (₹{prev_midpoint:,.1f}). "
                        f"Breakout momentum dampened."
                    )

            return {
                "retracement_healthy": is_healthy,
                "retest_support_50pct": retest_level,
                "candle_body_pts": round(candle_body if is_green_today else prev_body, 1),
                "retracement_ratio_pct": round(retracement_ratio, 1),
                "defense_desc": defense_desc,
                "invalidation_sl": invalidation_sl
            }
        else:
            # Bearish breakdown candle retracement check
            candle_body = max(last_o - last_c, 0.01)
            today_midpoint = round(last_o - (0.50 * candle_body), 2)
            lower_wick = max(last_c - last_l, 0.0)
            retracement_ratio = (lower_wick / (candle_body + 1e-6)) * 100.0

            is_healthy = (last_c <= today_midpoint) and (retracement_ratio <= 48.0)
            invalidation_sl = round(today_midpoint * 1.005, 1)

            if is_healthy:
                defense_desc = (
                    f"Breakdown Sustained: Price pinned below 50% breakdown candle body (Resistance: ₹{today_midpoint:,.1f}). "
                    f"Short sell on bounce to ₹{today_midpoint:,.1f} (SL: ₹{invalidation_sl:,.1f})"
                )
            else:
                defense_desc = f"Bear Trap Risk: Intraday recovery exceeded 50% of breakdown candle body."

            return {
                "retracement_healthy": is_healthy,
                "retest_support_50pct": today_midpoint,
                "candle_body_pts": round(candle_body, 1),
                "retracement_ratio_pct": round(retracement_ratio, 1),
                "defense_desc": defense_desc,
                "invalidation_sl": invalidation_sl
            }

    @staticmethod
    def evaluate_sector_tailwind(
        symbol: str,
        benchmark_dict: Optional[Dict[str, pd.DataFrame]],
        is_bullish: bool = True
    ) -> dict:
        """
        Determines if the parent sector index is providing a tailwind or headwind.
        """
        sector_name, bench_sym = MarketIntelAnalyzer.get_stock_sector(symbol)
        
        if not benchmark_dict:
            return {
                "sector_name": sector_name,
                "benchmark_symbol": bench_sym,
                "has_tailwind": True,  # Neutral default
                "tailwind_score_bonus": 5.0,
                "description": f"Sector: {sector_name}"
            }

        # Look for benchmark df
        bench_df = benchmark_dict.get(bench_sym)
        if bench_df is None or len(bench_df) < 20:
            bench_df = benchmark_dict.get("NIFTY")
            bench_sym = "NIFTY"

        if bench_df is None or len(bench_df) < 20:
            return {
                "sector_name": sector_name,
                "benchmark_symbol": bench_sym,
                "has_tailwind": True,
                "tailwind_score_bonus": 5.0,
                "description": f"Sector: {sector_name}"
            }

        last_b = bench_df.iloc[-1]
        prev_b = bench_df.iloc[-2]
        
        b_close = float(last_b['close'])
        b_change = ((b_close - float(prev_b['close'])) / float(prev_b['close'])) * 100.0
        
        # Sector moving average & RSI
        ema_20 = float(last_b.get('ema_20', b_close))
        rsi = float(last_b.get('rsi_14', 50.0))
        
        is_sector_bullish = (b_close >= ema_20) or (b_change > 0.1) or (rsi >= 52.0)
        is_sector_bearish = (b_close < ema_20) or (b_change < -0.1) or (rsi < 48.0)

        if is_bullish:
            has_tailwind = is_sector_bullish
            score_bonus = 10.0 if has_tailwind else 0.0
            status_text = "🌊 Active Sector Tailwind" if has_tailwind else "⚠️ Neutral/Counter Sector"
            desc = f"{status_text}: {sector_name} ({bench_sym} {b_change:+.2f}%, RSI {rsi:.0f})"
        else:
            has_tailwind = is_sector_bearish
            score_bonus = 10.0 if has_tailwind else 0.0
            status_text = "🌊 Downward Sector Pressure" if has_tailwind else "⚠️ Sector Resilient"
            desc = f"{status_text}: {sector_name} ({bench_sym} {b_change:+.2f}%)"

        return {
            "sector_name": sector_name,
            "benchmark_symbol": bench_sym,
            "has_tailwind": has_tailwind,
            "tailwind_score_bonus": score_bonus,
            "description": desc
        }

    @staticmethod
    def evaluate_panic_day_behavior(
        stock_df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame],
        lookback_days: int = 50
    ) -> dict:
        """
        Examines how the stock behaved on broad market 'Panic Days' (NIFTY -0.75% or worse).
        Institutional Accumulation Proof:
          - If a stock closed GREEN on a day NIFTY plunged, institutional demand absorbed all supply.
          - If a stock fell significantly less than NIFTY (Alpha >= +1.0%), it exhibits dominant Relative Strength.
        """
        if stock_df is None or nifty_df is None or len(stock_df) < 20 or len(nifty_df) < 20:
            return {
                "panic_resilient": False,
                "green_on_panic_days": 0,
                "panic_day_count": 0,
                "alpha_on_panic_days": 0.0,
                "score_bonus": 0.0,
                "description": "Panic-day data pending"
            }

        # Align series by date index
        s_ret = stock_df['close'].pct_change() * 100.0
        n_ret = nifty_df['close'].pct_change() * 100.0

        aligned = pd.DataFrame({"stock": s_ret, "nifty": n_ret}).dropna().iloc[-lookback_days:]

        # Identify NIFTY panic days: daily drop <= -0.75%
        panic_days = aligned[aligned['nifty'] <= -0.75]
        
        # Fallback if market had low volatility: take lowest 3 days
        if len(panic_days) < 2 and len(aligned) >= 10:
            panic_days = aligned.nsmallest(3, 'nifty')

        if len(panic_days) == 0:
            return {
                "panic_resilient": True,
                "green_on_panic_days": 0,
                "panic_day_count": 0,
                "alpha_on_panic_days": 0.0,
                "score_bonus": 5.0,
                "description": "Calm market conditions (Zero recent panic sessions)"
            }

        panic_count = len(panic_days)
        green_count = int((panic_days['stock'] > 0.0).sum())
        alpha_series = panic_days['stock'] - panic_days['nifty']
        avg_alpha = float(alpha_series.mean())

        # Recent panic day specifics
        last_panic = panic_days.iloc[-1]
        last_s_ret = float(last_panic['stock'])
        last_n_ret = float(last_panic['nifty'])

        is_resilient = (green_count >= 1) or (avg_alpha >= 1.25)
        score_bonus = 15.0 if green_count >= 1 else (10.0 if avg_alpha >= 1.0 else 0.0)

        if green_count >= 1:
            desc = (
                f"🛡️ Panic-Day Outperformer: Closed GREEN ({last_s_ret:+.2f}%) "
                f"during broad NIFTY selloff ({last_n_ret:+.2f}% drop). Smart money accumulation confirmed."
            )
        elif avg_alpha >= 1.0:
            desc = (
                f"🛡️ Resilient on Market Dips: Outperformed NIFTY by +{avg_alpha:.1f}% "
                f"across last {panic_count} panic sessions."
            )
        else:
            desc = f"Normal market correlation during panic days (Alpha: {avg_alpha:+.1f}%)"

        return {
            "panic_resilient": is_resilient,
            "green_on_panic_days": green_count,
            "panic_day_count": panic_count,
            "alpha_on_panic_days": round(avg_alpha, 2),
            "score_bonus": score_bonus,
            "description": desc
        }

    @staticmethod
    def compute_mansfield_rs(
        stock_df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame],
        period: int = 50
    ) -> Tuple[float, str]:
        """
        Computes Mansfield Relative Strength (MRS) vs NIFTY 50.
        MRS = ((RS / SMA(RS, 50)) - 1) * 100
        Values > 0 indicate institutional outperformance regime (Stage-2 leader).
        """
        if stock_df is None or nifty_df is None or len(stock_df) < period or len(nifty_df) < period:
            return 0.0, "RS Neutral"

        s_close = stock_df['close']
        n_close = nifty_df['close']

        aligned = pd.DataFrame({"s": s_close, "n": n_close}).dropna()
        if len(aligned) < period:
            return 0.0, "RS Neutral"

        rs = aligned['s'] / aligned['n']
        rs_sma = rs.rolling(period).mean()
        mrs_series = ((rs / rs_sma) - 1.0) * 100.0

        latest_mrs = float(mrs_series.iloc[-1])
        if latest_mrs >= 2.0:
            rs_trend = f"Strong Leader (MRS: +{latest_mrs:.1f}%)"
        elif latest_mrs > 0:
            rs_trend = f"Outperforming (MRS: +{latest_mrs:.1f}%)"
        else:
            rs_trend = f"Lagging Benchmark (MRS: {latest_mrs:.1f}%)"

        return round(latest_mrs, 2), rs_trend


# Backward-compatibility aliases
SectorManager = MarketIntelAnalyzer
MansfieldRelativeStrength = MarketIntelAnalyzer
