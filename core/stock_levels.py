"""
Custom Stock Two-Way Action Levels Analyzer.
Calculates institutional key price thresholds for any equity or commodity:
1. Level ABOVE which stock can RISE (Upside Momentum Breakout Trigger) + Upside Targets & SL
2. Level BELOW which stock can FALL (Downside Breakdown Floor) + Downside Targets & SL
3. Neutral Consolidation / Chop Zone between the two triggers
4. Tactical Trade Thesis and Options Setup for actionable execution
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.relative_strength import MarketIntelAnalyzer
from core.universe import UniverseManager


class StockLevelAnalyzer:
    """
    Evaluates any stock or commodity to identify institutional two-way breakout/breakdown levels:
    - Level above which stock can rise with explosive upward continuation
    - Level below which stock can fall with accelerating downward selling
    """

    @staticmethod
    def _determine_strike_step(price: float) -> int:
        """Determines typical NSE option strike step based on underlying stock price."""
        if price < 150:
            return 5
        elif price < 350:
            return 10
        elif price < 750:
            return 20
        elif price < 2000:
            return 50
        elif price < 5000:
            return 100
        else:
            return 100

    @classmethod
    def analyze_stock(
        cls,
        symbol: str,
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> Optional[dict]:
        """
        Analyzes a single stock's OHLCV history to produce two-way action levels and trade triggers.
        """
        if df is None or len(df) < 20:
            return None

        clean = UniverseManager.to_clean_symbol(symbol)
        df_ind = enrich_with_indicators(df.copy())
        last = df_ind.iloc[-1]
        prev = df_ind.iloc[-2]

        close = float(last['close'])
        open_p = float(last['open'])
        high = float(last['high'])
        low = float(last['low'])
        volume = float(last.get('volume', 0.0))
        prev_close = float(prev['close'])

        change_pct = ((close - prev_close) / (prev_close + 1e-6)) * 100.0

        atr = float(last.get('atr_14', close * 0.02))
        rsi = float(last.get('rsi_14', 50.0))
        rvol = float(last.get('rvol_20', 1.0))
        ema20 = float(last.get('ema_20', close))
        ema50 = float(last.get('ema_50', close))
        sma200 = float(last.get('sma_200', ema50)) if 'sma_200' in df_ind.columns and not np.isnan(last['sma_200']) else ema50

        # Multi-session swing pivots
        lookback_len = min(252, len(df_ind))
        high_52w = float(df_ind['high'].iloc[-lookback_len:].max())
        low_52w = float(df_ind['low'].iloc[-lookback_len:].min())

        prev_20_high = float(df_ind['high'].iloc[-21:-1].max()) if len(df_ind) >= 21 else high
        prev_20_low = float(df_ind['low'].iloc[-21:-1].min()) if len(df_ind) >= 21 else low

        # Classical Floor Pivots
        pivot = (high + low + close) / 3.0
        r1 = (2.0 * pivot) - low
        s1 = (2.0 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        # ----------------------------------------------------------------------
        # 1. LEVEL ABOVE WHICH STOCK CAN RISE (BULLISH BREAKOUT TRIGGER)
        # ----------------------------------------------------------------------
        # A stock breaks out to the upside when it overcomes session supply (today's high),
        # classic floor resistance (R1), or nearby multi-day swing pivot resistance.
        if close >= high * 0.998:
            # Already pushing session high: trigger is sustained continuation above current high / R1
            rise_above = round(max(high, r1 * 0.999), 2)
        else:
            # Trading below high: breakout triggers upon clearing session high or R1
            rise_above = round(max(high, r1), 2)

        # If 20-day swing high is nearby within 1.5% overhead, factor it into the resistance ceiling
        if 0 < (prev_20_high - close) / close <= 0.02:
            rise_above = round(max(rise_above, prev_20_high), 2)

        upside_t1 = round(rise_above + (0.80 * atr), 2)
        upside_t2 = round(rise_above + (1.60 * atr), 2)
        upside_t3 = round(rise_above + (2.50 * atr), 2)
        upside_sl = round(rise_above - (0.50 * atr), 2)
        upside_risk = max(rise_above - upside_sl, 0.05)
        upside_reward = upside_t1 - rise_above
        upside_rr = f"1:{upside_reward / upside_risk:.1f}"

        # ----------------------------------------------------------------------
        # 2. LEVEL BELOW WHICH STOCK CAN FALL (BEARISH BREAKDOWN FLOOR)
        # ----------------------------------------------------------------------
        # A stock breaks down to the downside when it cracks session buyer absorption (session low),
        # classic floor support (S1), or nearby multi-day swing pivot support.
        if close <= low * 1.002:
            # Already probing session low: breakdown triggers upon breach below low / S1
            fall_below = round(min(low, s1 * 1.001), 2)
        else:
            fall_below = round(min(low, s1), 2)

        # If 20-day swing low is nearby within 1.5% below, incorporate it as the critical floor
        if 0 < (close - prev_20_low) / close <= 0.02:
            fall_below = round(min(fall_below, prev_20_low), 2)

        downside_t1 = round(fall_below - (0.80 * atr), 2)
        downside_t2 = round(fall_below - (1.60 * atr), 2)
        downside_t3 = round(fall_below - (2.50 * atr), 2)
        downside_sl = round(fall_below + (0.50 * atr), 2)
        downside_risk = max(downside_sl - fall_below, 0.05)
        downside_reward = fall_below - downside_t1
        downside_rr = f"1:{downside_reward / downside_risk:.1f}"

        # ----------------------------------------------------------------------
        # 3. CONSOLIDATION / NO-TRADE ZONE & CURRENT POSITION
        # ----------------------------------------------------------------------
        chop_zone = f"₹{fall_below:,.2f} — ₹{rise_above:,.2f}"
        dist_rise = ((rise_above - close) / close) * 100.0
        dist_fall = ((close - fall_below) / close) * 100.0

        # State determination
        if close >= rise_above:
            status = "🟢 ACTIVE EXPANSION (Trading Above Rise Level)"
            status_color = "#00E676"
            bias = "BULLISH"
        elif close <= fall_below:
            status = "🔴 ACTIVE BREAKDOWN (Trading Below Fall Level)"
            status_color = "#FF5252"
            bias = "BEARISH"
        elif dist_rise <= 0.8:
            status = "⚡ TESTING RESISTANCE (Near Upside Rise Level)"
            status_color = "#64B5F6"
            bias = "BULLISH_BIAS"
        elif dist_fall <= 0.8:
            status = "⚠️ TESTING SUPPORT (Near Downside Fall Level)"
            status_color = "#FFA726"
            bias = "BEARISH_BIAS"
        else:
            status = "🟡 CONSOLIDATING (Inside Neutral Chop Range)"
            status_color = "#FFD54F"
            bias = "NEUTRAL"

        # Directional Trade Thesis
        sector = MarketIntelAnalyzer.get_sector(clean)
        is_comm = UniverseManager.is_commodity(clean)
        unit = "pts" if is_comm else "₹"

        thesis_rise = (
            f"🟢 RALLY TRIGGER: Sustained hold ABOVE ₹{rise_above:,.2f} confirms buyer absorption and triggers momentum rally "
            f"toward Targets ₹{upside_t1:,.2f} (T1) and ₹{upside_t2:,.2f} (T2) with Invalidation Stop Loss at ₹{upside_sl:,.2f} (R:R {upside_rr})."
        )
        thesis_fall = (
            f"🔴 DOWNSIDE TRIGGER: Crack and breach BELOW ₹{fall_below:,.2f} invalidates demand and accelerates downward selling "
            f"toward Targets ₹{downside_t1:,.2f} (T1) and ₹{downside_t2:,.2f} (T2) with Invalidation Stop Loss at ₹{downside_sl:,.2f} (R:R {downside_rr})."
        )
        thesis_neutral = (
            f"🟡 NO-TRADE ZONE: Currently trading at ₹{close:,.2f} inside range ({chop_zone}). "
            f"Wait for a decisive breakout above ₹{rise_above:,.2f} or breakdown below ₹{fall_below:,.2f} before initiating high-conviction positions."
        )

        full_thesis = f"{thesis_rise} | {thesis_fall}"

        # ----------------------------------------------------------------------
        # 4. OPTIONS EXPLORATION (CE / PE PLAYS)
        # ----------------------------------------------------------------------
        step = cls._determine_strike_step(close)
        atm_strike = int(round(close / step) * step)
        call_strike = atm_strike if close < rise_above else atm_strike + step
        put_strike = atm_strike if close > fall_below else atm_strike - step

        options_setup = {
            "is_fno": UniverseManager.is_fno(clean),
            "strike_step": step,
            "atm_strike": atm_strike,
            "call_option_play": f"{clean} {call_strike} CE (Upside Trigger)",
            "put_option_play": f"{clean} {put_strike} PE (Downside Trigger)",
        }

        # Mansfield RS vs Benchmark if provided
        mrs_val = 0.0
        if benchmark_df is not None and len(benchmark_df) >= 20:
            try:
                mrs_val, _ = MarketIntelAnalyzer.compute_mansfield_rs(df, benchmark_df, period=50)
            except Exception:
                pass

        capture_time = df.attrs.get("capture_time", "Live")
        data_source = df.attrs.get("data_source", "Market Feed")

        return {
            "symbol": clean,
            "display_name": f"{clean} ({sector})",
            "sector": sector,
            "close": round(close, 2),
            "open": round(open_p, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "rvol": round(rvol, 2),
            "rsi": round(rsi, 1),
            "atr": round(atr, 2),
            "capture_time": capture_time,
            "data_source": data_source,
            
            # Action Levels
            "rise_above": rise_above,
            "rise_distance_pct": round(dist_rise, 2),
            "upside_target_1": upside_t1,
            "upside_target_2": upside_t2,
            "upside_target_3": upside_t3,
            "upside_stop_loss": upside_sl,
            "upside_risk_reward": upside_rr,
            
            "fall_below": fall_below,
            "fall_distance_pct": round(dist_fall, 2),
            "downside_target_1": downside_t1,
            "downside_target_2": downside_t2,
            "downside_target_3": downside_t3,
            "downside_stop_loss": downside_sl,
            "downside_risk_reward": downside_rr,
            
            "chop_zone": chop_zone,
            "status": status,
            "status_color": status_color,
            "bias": bias,
            "thesis_rise": thesis_rise,
            "thesis_fall": thesis_fall,
            "thesis_neutral": thesis_neutral,
            "trade_thesis": full_thesis,
            
            # Trend context
            "ema_20": round(ema20, 2),
            "ema_50": round(ema50, 2),
            "sma_200": round(sma200, 2),
            "is_above_200sma": close >= sma200,
            "is_stage2_uptrend": (close >= ema20 >= ema50),
            "is_stage4_downtrend": (close <= ema20 <= ema50),
            "prev_20_high": round(prev_20_high, 2),
            "prev_20_low": round(prev_20_low, 2),
            "52w_high": round(high_52w, 2),
            "52w_low": round(low_52w, 2),
            "mansfield_rs": round(mrs_val, 2),
            
            # Pivots
            "pivots": {
                "pivot": round(pivot, 2),
                "r1": round(r1, 2),
                "r2": round(r2, 2),
                "s1": round(s1, 2),
                "s2": round(s2, 2)
            },
            "options_setup": options_setup
        }

    @classmethod
    def analyze_batch(
        cls,
        data: Dict[str, pd.DataFrame],
        benchmark_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[dict]:
        """
        Analyzes a batch of stock OHLCVs to return action levels for all symbols.
        """
        nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)

        results = []
        for sym, df in data.items():
            if df is not None and len(df) >= 20:
                res = cls.analyze_stock(sym, df, benchmark_df=nifty_df)
                if res:
                    results.append(res)

        # Sort: stocks near breakout/breakdown triggers first
        results.sort(key=lambda x: min(abs(x["rise_distance_pct"]), abs(x["fall_distance_pct"])))
        return results
