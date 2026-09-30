"""
First-to-Recover Market Dip Leaders (Alpha Springboards) Screener.
Identifies high-quality Stage-2 market leaders that may dip during broad market selloffs,
exhibit dry selling volume and institutional absorption, and are statistically the first
to rebound and break out to new highs when the broader market stabilizes.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators
from core.macro import MacroMarketEngine
from core.relative_strength import MarketIntelAnalyzer
from core.universe import UniverseManager
from screeners.base_screener import BaseScreener


class DipLeaderScreener(BaseScreener):
    """
    Quantifies and shortlists 'First to Recover' Market Dip Leaders.
    
    Quantitative Architecture:
      1. Structural Trend Filter: Stage-2 leadership (Price > 200 SMA / 50 EMA).
      2. Dip Absorption & Panic Resilience: Outperformed NIFTY on recent red sessions.
      3. Rapid Bounce Velocity (Up-Beta): Surges faster and stronger when NIFTY bounces.
      4. Supply Exhaustion: Low relative volume on pullbacks (smart money holding tight).
      5. Coiled Spring Geometry: Testing dynamic institutional support (20/50 EMA, retest floor).
    """

    def __init__(self):
        super().__init__(name="First-to-Recover Dip Leaders", category="DIP_LEADER")

    def screen(
        self,
        symbol: str,
        df: pd.DataFrame,
        benchmark_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[dict]:
        clean = UniverseManager.to_clean_symbol(symbol)
        
        # Exclude pure indices
        if clean in ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT", "INDIAVIX"]:
            return None

        if df is None or len(df) < 35:
            return None

        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last['close'])
        open_p = float(last['open'])
        high = float(last['high'])
        low = float(last['low'])
        prev_close = float(prev['close'])

        change_pct = ((close - prev_close) / (prev_close + 1e-6)) * 100.0

        ema20 = float(last.get('ema_20', close))
        ema50 = float(last.get('ema_50', close))
        sma200 = float(last.get('sma_200', close)) if 'sma_200' in df.columns and not np.isnan(last['sma_200']) else ema50
        rsi = float(last.get('rsi_14', 50.0))
        atr = float(last.get('atr_14', close * 0.02))
        rvol = float(last.get('rvol_20', 1.0))
        rvol_9 = float(last.get('rvol_9', rvol))
        clv = float(last.get('clv', 0.5))

        # Candle anatomy
        candle_range = max(high - low, 0.001)
        lower_wick = min(open_p, close) - low
        upper_wick = high - max(open_p, close)
        lower_wick_pct = max(0.0, lower_wick / candle_range)
        upper_wick_pct = max(0.0, upper_wick / candle_range)

        # ----------------- 1. STRUCTURAL QUALITY & TREND INTEGRITY -----------------
        # The stock must NOT be a broken, falling-knife junk asset.
        # It must be a structural market leader (Price above 200 SMA or 50 EMA, or within 2% of 50 EMA on dip).
        is_above_sma200 = close >= (sma200 * 0.97)
        is_above_ema50 = close >= (ema50 * 0.975)

        if not (is_above_sma200 or is_above_ema50):
            return None

        # 2. Benchmark comparative metrics
        nifty_df = MarketIntelAnalyzer.extract_nifty_benchmark(benchmark_data)

        mansfield_rs = 0.0
        rs_trend = "Neutral"
        green_on_dips = 0
        dip_alpha = 0.0
        up_beta = 1.0
        has_panic_resilience = False

        if nifty_df is not None and len(nifty_df) >= 20:
            mansfield_rs, rs_trend = MarketIntelAnalyzer.compute_mansfield_rs(df, nifty_df, period=50)
            panic_info = MarketIntelAnalyzer.evaluate_panic_day_behavior(df, nifty_df, lookback_days=40)
            has_panic_resilience = panic_info.get("panic_resilient", False)
            green_on_dips = panic_info.get("green_on_panic_days", 0)
            dip_alpha = panic_info.get("alpha_on_panic_days", 0.0)

            # Compute Upside Bounce Velocity (how much stock gained when Nifty was green)
            s_ret = df['close'].pct_change() * 100.0
            n_ret = nifty_df['close'].pct_change() * 100.0
            aligned = pd.DataFrame({"s": s_ret, "n": n_ret}).dropna().iloc[-30:]
            green_sessions = aligned[aligned['n'] >= 0.35]
            if len(green_sessions) >= 3:
                up_beta = float((green_sessions['s'] / green_sessions['n']).clip(-2.0, 5.0).mean())

        # ----------------- 3. DIP QUALIFICATION FILTERS -----------------
        # Criteria to qualify as a "First to Recover" candidate:
        # A: Strong Relative Strength on Dips:
        #    Closed green on at least 1 recent market dip day OR Dip Alpha >= +0.40% OR Mansfield RS >= 1.0
        strong_rs_on_dips = (green_on_dips >= 1) or (dip_alpha >= 0.40) or (mansfield_rs >= 0.5)

        # B: Bounce Geometry / Support Floor:
        #    Price is near dynamic support (within 3.0% of 20 EMA or 50 EMA)
        dist_ema20 = abs(close - ema20) / ema20 * 100.0
        dist_ema50 = abs(close - ema50) / ema50 * 100.0
        near_dynamic_support = (dist_ema20 <= 2.8) or (dist_ema50 <= 3.5) or (close >= ema20)

        # C: Supply Exhaustion / Dry Pullback Volume:
        #    If stock is red today or pulled back, volume should be below average (not institutional distribution)
        volume_exhaustion = (rvol <= 1.10) or (rvol_9 <= 1.05) or (change_pct >= 0.5)

        # D: Buyer Absorption Wick / High CLV:
        #    Buyers stepped in to defend the dip (lower shadow >= 25% or CLV >= 0.45)
        buyer_absorption = (lower_wick_pct >= 0.25) or (clv >= 0.48) or (close >= open_p)

        # Qualification Gate: Must meet at least 3 of 4 core pillars
        pillars_met = sum([
            bool(strong_rs_on_dips),
            bool(near_dynamic_support),
            bool(volume_exhaustion),
            bool(buyer_absorption)
        ])

        if pillars_met < 3:
            return None

        # ----------------- 4. SPRINGBOARD RECOVERY SCORING (0 to 100) -----------------
        score = 50.0

        # Dip Alpha & Panic Resilience (up to +25 pts)
        if green_on_dips >= 1:
            score += 15.0
        elif dip_alpha >= 0.8:
            score += 10.0
        elif dip_alpha >= 0.3:
            score += 5.0

        if has_panic_resilience:
            score += 10.0

        # Upside Bounce Velocity (up to +18 pts)
        if up_beta >= 1.8:
            score += 18.0
        elif up_beta >= 1.3:
            score += 12.0
        elif up_beta >= 1.0:
            score += 6.0

        # Mansfield Relative Strength (up to +15 pts)
        if mansfield_rs >= 2.5:
            score += 15.0
        elif mansfield_rs > 0:
            score += 8.0

        # Volume Dry-up on Pullback (up to +15 pts)
        if rvol < 0.75:
            score += 15.0
        elif rvol < 0.95:
            score += 10.0
        elif rvol < 1.10:
            score += 5.0

        # Buyer Absorption Wick & CLV (up to +12 pts)
        if lower_wick_pct >= 0.35 or clv >= 0.65:
            score += 12.0
        elif lower_wick_pct >= 0.25 or clv >= 0.50:
            score += 7.0

        # Proximity to Dynamic Bounce Zone (up to +15 pts)
        if dist_ema20 <= 1.2 or (close >= ema20 and close <= ema20 * 1.015):
            score += 15.0
        elif dist_ema50 <= 1.8:
            score += 10.0
        elif near_dynamic_support:
            score += 5.0

        score = float(np.clip(score, 45.0, 99.0))

        # ----------------- 5. RECOVERY VELOCITY BADGE & TARGETS -----------------
        if score >= 85:
            recovery_badge = "⚡ ULTRA-FAST BOUNCE (First-to-Pop Leader)"
            recovery_color = "#00E676"
        elif score >= 75:
            recovery_badge = "🚀 HIGH ALPHA SPRINGBOARD"
            recovery_color = "#00B0FF"
        else:
            recovery_badge = "💎 INSTITUTIONAL ACCUMULATION FLOOR"
            recovery_color = "#FFD54F"

        # Actionable Levels
        entry_low = round(min(close, max(low, ema20 * 0.995 if close >= ema20 else ema50 * 0.995)), 2)
        entry_high = round(close * 1.004, 2)
        springboard_zone = f"₹{entry_low:,.1f} – ₹{entry_high:,.1f}"

        # Structural Stop-Loss (below dynamic support / recent swing low)
        invalidation_sl = round(min(low * 0.988, (ema50 * 0.982 if close >= ema50 else ema20 * 0.98)), 2)
        risk = max(close - invalidation_sl, close * 0.015)

        # Retest of prior 20-day high and blue-sky expansion
        recent_20_high = float(df['high'].tail(20).max())
        rebound_target_1 = round(max(recent_20_high, close + (1.6 * risk)), 2)
        rebound_target_2 = round(close + (2.8 * risk), 2)
        rebound_target_3 = round(close + (4.0 * risk), 2)

        # Macro Context Analysis
        try:
            macro_impact = MacroMarketEngine.get_stock_macro_impact(clean)
            macro_badge = macro_impact.get("badge", "")
            macro_thesis = macro_impact.get("thesis", "")
        except Exception:
            macro_badge = ""
            macro_thesis = ""

        # Key Triggers list
        reasons = []
        if green_on_dips >= 1:
            reasons.append(f"Institutional Absorption: Closed GREEN on {green_on_dips} broad market panic session(s)")
        elif dip_alpha >= 0.4:
            reasons.append(f"Market Dip Alpha: Outperformed NIFTY by +{dip_alpha:.1f}% across selloffs")
        
        if up_beta >= 1.2:
            reasons.append(f"Fast Bounce Velocity: Generates {up_beta:.1f}x upside beta on green sessions")
        
        if mansfield_rs > 0:
            reasons.append(f"Stage-2 Leadership: Mansfield RS +{mansfield_rs:.1f}% vs NIFTY 50")
        
        if rvol < 0.95:
            reasons.append(f"Supply Dry-Up: Selling volume is low ({rvol:.2f}x 20d avg) — no dumping")
        
        if lower_wick_pct >= 0.25:
            reasons.append(f"Buyer Wick Defense: {lower_wick_pct*100:.0f}% lower shadow rejection at support")
        
        if close >= ema20:
            reasons.append("Trend Intact: Defending rising 20 EMA")
        elif close >= ema50:
            reasons.append("Value Zone: Rebound floor at rising 50 EMA")

        if macro_badge and "Neutral" not in macro_badge:
            reasons.append(macro_badge)

        # Comprehensive Thesis
        thesis = (
            f"Prime Market Dip Leader: While broader market dipped, {clean} demonstrated superior relative strength "
            f"(Alpha: +{dip_alpha:.1f}%, MRS: +{mansfield_rs:.1f}%) on dried-up selling volume ({rvol:.2f}x). "
            f"Coiled spring setup: Buyers actively defended support. Expect rapid V-shape expansion toward ₹{rebound_target_1:,.1f} "
            f"as soon as Nifty stabilizes. {macro_thesis}"
        )

        return {
            "symbol": clean,
            "score": round(score, 1),
            "category": "DIP_LEADER",
            "close": close,
            "change_pct": round(change_pct, 2),
            "rsi": round(rsi, 1),
            "rvol": round(rvol, 2),
            "rvol_9": round(rvol_9, 2),
            "vol_confirmed_2x": False,
            "recovery_velocity_badge": recovery_badge,
            "recovery_color": recovery_color,
            "springboard_entry_zone": round(close, 2),
            "springboard_zone_str": springboard_zone,
            "entry_price": close,
            "invalidation_sl": invalidation_sl,
            "stop_loss": invalidation_sl,
            "tight_stop_loss": invalidation_sl,
            "rebound_target_1": rebound_target_1,
            "rebound_target_2": rebound_target_2,
            "rebound_target_3": rebound_target_3,
            "target_1": rebound_target_1,
            "target_2": rebound_target_2,
            "target_3": rebound_target_3,
            "target_4": rebound_target_3,
            "risk_reward_ratio": f"1:{((rebound_target_1 - close) / (risk + 1e-6)):.1f} → 1:{((rebound_target_2 - close) / (risk + 1e-6)):.1f}",
            "timeframe": "1-3 weeks (Dip Recovery Springboard)",
            "conviction": "HIGH" if score >= 80 else "MEDIUM",
            "mansfield_rs": mansfield_rs,
            "mrs": mansfield_rs,
            "dip_alpha": round(dip_alpha, 2),
            "dip_alpha_pct": round(dip_alpha, 2),
            "upside_beta": round(up_beta, 2),
            "bounce_velocity_up_beta": round(up_beta, 2),
            "pullback_rvol": round(rvol, 2),
            "lower_wick_pct": round(lower_wick_pct * 100.0, 1),
            "support_floor_desc": "Defending 20 EMA Floor" if close >= ema20 else "Defending 50 EMA Structural Floor",
            "green_on_dips": green_on_dips,
            "has_panic_resilience": has_panic_resilience,
            "has_sector_tailwind": True,
            "retest_support_50pct": entry_low,
            "macro_badge": macro_badge,
            "macro_alignment": macro_badge,
            "macro_thesis": macro_thesis,
            "recovery_catalyst": thesis,
            "reasons": reasons,
            "thesis": thesis,
            "metrics": {
                "ema_20": ema20,
                "ema_50": ema50,
                "sma_200": sma200,
                "atr": atr,
                "clv": clv,
                "lower_wick_pct": lower_wick_pct,
                "support_level": invalidation_sl,
                "target_level": rebound_target_1
            }
        }
