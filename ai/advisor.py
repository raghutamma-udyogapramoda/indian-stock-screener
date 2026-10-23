"""
AI Trade Advisor powered by Google Gemini.
Evaluates shortlisted quantitative setups and generates actionable trade plans.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from ai.prompt_builder import PromptBuilder
from config.settings import GEMINI_API_KEY, GEMINI_MODEL
from core.macro import MacroMarketEngine


class AIAdvisor:
    """Orchestrates LLM trade evaluations using Gemini Flash with graceful algorithmic fallback."""

    def __init__(self, api_key: Optional[str] = None, model: str = GEMINI_MODEL):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        if not self.api_key or self.api_key.startswith("your_"):
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception:
            self.client = None

    @property
    def is_ai_ready(self) -> bool:
        return self.client is not None

    def analyze_candidates(
        self,
        candidates: List[Dict[str, Any]],
        macro_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Passes shortlisted setups and macro context to Gemini.
        If API key is missing or calls fail, generates an algorithmic trade plan so workflow is uninterrupted.
        """
        if not candidates:
            return []

        if macro_context is None:
            try:
                macro_context = MacroMarketEngine.analyze_macro_regime()
            except Exception:
                macro_context = None

        # If live AI is available, query Gemini
        if self.is_ai_ready:
            try:
                user_content = PromptBuilder.build_user_prompt(candidates, macro_context=macro_context)
                prompt_full = f"{PromptBuilder.SYSTEM_PROMPT}\n\n{user_content}"
                
                # Try preferred model and fallback if needed
                models_to_try = [self.model]
                if self.model not in ["gemini-2.0-flash", "gemini-1.5-flash"]:
                    models_to_try.extend(["gemini-2.0-flash", "gemini-1.5-flash"])

                response = None
                last_err = None
                for m in models_to_try:
                    try:
                        from google.genai import types
                        config = types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        )
                        response = self.client.models.generate_content(
                            model=m,
                            contents=prompt_full,
                            config=config,
                        )
                        if response and response.text:
                            break
                    except Exception as err:
                        last_err = err
                        continue

                if response and response.text:
                    text = response.text.strip()
                    # Clean any markdown code blocks
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    
                    parsed = json.loads(text.strip())
                    if isinstance(parsed, list) and len(parsed) > 0:
                        cand_map = {c.get("symbol"): c for c in candidates if c.get("symbol")}
                        for p in parsed:
                            sym_match = cand_map.get(p.get("symbol"))
                            if sym_match:
                                if "close" not in p:
                                    p["close"] = float(sym_match.get("close", 0.0))
                                if "change_pct" not in p:
                                    p["change_pct"] = float(sym_match.get("change_pct", 0.0))
                        # Enforce mathematical progression and anti-inversion validation on LLM output
                        guarded_plans = [self._sanitize_and_guard_plan(p, cand_map.get(p.get("symbol"))) for p in parsed]
                        return guarded_plans
            except Exception:
                # Smooth fallback to algorithmic precision engine
                pass

        # Algorithmic fallback: generates mathematically sound trade plan
        return self._generate_algorithmic_trade_plan(candidates, macro_context=macro_context)

    def _generate_algorithmic_trade_plan(
        self,
        candidates: List[Dict[str, Any]],
        macro_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates mathematically precise Entry, Stop-Loss, and Target levels.
        Integrates macroeconomic forces (Crude Oil, USD/INR, VIX, NIFTY) and Dip Leaders.
        Serves as the high-precision algorithmic engine (and zero-dependency fallback when LLM is offline).

        Formula & Risk Architecture per Category:
          - BTST:
              * Entry: Close price (accumulate in final minutes).
              * Stop-Loss: Close - (1.2 * ATR_14).
              * Target 1: Close + (1.5 * ATR_14) [Capture opening gap-up].
              * Target 2: Close + (2.5 * ATR_14).
          - BREAKOUT:
              * Entry: Close * 1.002 (momentum tick above breakout level).
              * Stop-Loss: 20 EMA or Close - (2.0 * ATR_14).
              * Target 1: Entry + (2.0 * Risk) [1:2.0 R:R].
              * Target 2: Entry + (3.5 * Risk) [1:3.5 R:R].
          - BREAKDOWN (Short):
              * Entry: Close * 0.998 (continuation tick below breakdown pivot).
              * Stop-Loss: 20 EMA or Close + (1.8 * ATR_14).
              * Target 1: Entry - (1.8 * Risk).
              * Target 2: Entry - (3.0 * Risk).
          - SWING:
              * Entry: Close price at dynamic support (EMA 20/50 / lower BB).
              * Stop-Loss: Recent 3-bar swing low or Close - (1.6 * ATR_14).
              * Target 1: 20-day swing resistance or Entry + (2.5 * Risk).
              * Target 2: Entry + (3.8 * Risk) [Asymmetric 1:2.5+ R:R].
          - INTRADAY (Long / Short):
              * Stop-Loss: Prior day low (for Long) or day high (for Short).
              * Targets: Scaled by 1.5x and 2.5x of intra-session risk.
          - OPTIONS (CE / PE):
              * Action: BUY_CE or BUY_PE.
              * Strike: Auto-selected ATM strike with ATR volatility targets.
        """
        plans = []
        for c in candidates:
            cat = c.get("category", "BREAKOUT").upper()
            close = float(c.get("close", 100.0))
            metrics = c.get("metrics", {})
            atr = float(metrics.get("atr", close * 0.02))

            vol_conf = c.get("vol_confirmed_2x", False)
            vol_mult = c.get("rvol_9", c.get("rvol", 1.0))
            vol_tag = f" Institutional volume confirmed ({vol_mult:.1f}x 9-day avg)." if vol_conf else ""

            # Retracement Defense Level
            retest_lvl = float(c.get("retest_support_50pct", metrics.get("retest_support_50pct", 0.0)))
            retest_tag = f" 50% candle body defense at ₹{retest_lvl:,.1f}." if retest_lvl > 0 else ""

            # Sector Tailwind & Panic-Day Resilience
            has_tailwind = c.get("has_sector_tailwind", False)
            sector_name = c.get("sector", "Sector")
            tailwind_tag = f" Supported by {sector_name} tailwind." if has_tailwind else ""
            
            panic_res = c.get("panic_day_resilient", False)
            panic_tag = " Smart money absorption (held green/outperformed on NIFTY panic days)." if panic_res else ""

            extra_insight = f"{retest_tag}{tailwind_tag}{panic_tag}"

            if cat == "BTST":
                entry = close
                tight_sl = round(close - (0.8 * atr), 2)
                cons_sl = round(close - (1.2 * atr), 2)
                stop_loss = cons_sl
                risk = max(entry - tight_sl, close * 0.01)
                target_1 = round(close + (1.0 * atr), 2)
                target_2 = round(close + (1.8 * atr), 2)
                target_3 = round(close + (2.5 * atr), 2)
                target_4 = round(close + (3.5 * atr), 2)
                action = "BUY"
                timeframe = "1-2 days (Overnight BTST)"
                conviction = "HIGH" if (c.get("score", 0) >= 80 or vol_conf or panic_res) else "MEDIUM"
                thesis = f"Strong closing accumulation with {vol_mult:.1f}x volume.{vol_tag}{extra_insight} Target open gap-up."
                trailing_playbook = f"Book 50% on morning gap-up at T1/T2 (₹{target_1:,.1f}-₹{target_2:,.1f}) and move SL to entry."

            elif cat == "BREAKOUT":
                entry = round(close * 1.002, 2)
                tight_sl = round(retest_lvl * 0.995 if retest_lvl > 0 else entry - (1.1 * atr), 2)
                cons_sl = round(float(metrics.get("ema_20", entry - (1.8 * atr))), 2)
                if tight_sl >= entry:
                    tight_sl = round(entry - (1.1 * atr), 2)
                stop_loss = tight_sl
                risk = max(entry - stop_loss, close * 0.015)
                target_1 = round(entry + (1.25 * risk), 2)
                target_2 = round(entry + (2.0 * risk), 2)
                target_3 = round(entry + (2.9 * risk), 2)
                recent_20_h = float(metrics.get("target_level", entry + (4.0 * risk)))
                target_4 = round(max(recent_20_h, entry + (4.0 * risk)), 2)
                action = "BUY"
                timeframe = "2-4 weeks (Positional Breakout)"
                conviction = "HIGH" if (c.get("score", 0) >= 75 or vol_conf or has_tailwind) else "MEDIUM"
                thesis = f"Stage-2 breakout coiling near key resistance.{vol_tag}{extra_insight}"
                trailing_playbook = f"Book 33% at T1 (₹{target_1:,.1f}) & trail SL to Cost (₹{entry:,.1f}). Book next 33% at T2 (₹{target_2:,.1f}) & trail SL to T1. Let remaining 34% runners target T3 & T4."

            elif cat == "BREAKDOWN":
                entry = round(close * 0.998, 2)
                tight_sl = round(retest_lvl * 1.005 if retest_lvl > 0 else entry + (1.1 * atr), 2)
                cons_sl = round(float(metrics.get("ema_20", entry + (1.8 * atr))), 2)
                if tight_sl <= entry:
                    tight_sl = round(entry + (1.1 * atr), 2)
                stop_loss = tight_sl
                risk = max(stop_loss - entry, close * 0.015)
                target_1 = round(entry - (1.25 * risk), 2)
                target_2 = round(entry - (2.0 * risk), 2)
                target_3 = round(entry - (2.9 * risk), 2)
                target_4 = round(entry - (4.0 * risk), 2)
                action = "SELL"
                timeframe = "1-3 weeks (Bearish Short)"
                conviction = "HIGH" if (c.get("score", 0) >= 75 or vol_conf) else "MEDIUM"
                thesis = f"Stage-4 structural breakdown below pivot support with {vol_mult:.1f}x selling volume.{vol_tag}{tailwind_tag}"
                trailing_playbook = f"Cover 33% short at T1 (₹{target_1:,.1f}) & trail SL to Entry. Cover next 33% at T2 (₹{target_2:,.1f}) & trail SL to T1. Hold runners for T3 & T4."

            elif cat == "SWING":
                entry = close
                # Base SL: 1.2x ATR (matches active swing trader base low)
                tight_sl = round(entry - (1.2 * atr), 2)
                # Structural SL: 1.5x ATR or multi-bar swing support floor
                cons_sl = round(min(float(metrics.get("support_level", entry - (1.5 * atr))), entry - (1.5 * atr)), 2)
                
                # Active traders anchor risk to base SL (e.g. ₹2,424 on RRKABEL)
                stop_loss = tight_sl
                risk = max(entry - stop_loss, close * 0.015)

                # Institutional Staggered Target Ladder (1.2R, 1.7R, 2.45R, 3.6R / 52W High)
                target_1 = round(entry + (1.2 * risk), 2)
                target_2 = round(entry + (1.7 * risk), 2)
                target_3 = round(entry + (2.45 * risk), 2)
                recent_20_h = float(metrics.get("target_level", entry + (3.6 * risk)))
                target_4 = round(max(recent_20_h, entry + (3.6 * risk)), 2)

                action = "BUY"
                timeframe = "1-4 weeks (Swing Pullback)"
                conviction = "HIGH" if (c.get("score", 0) >= 75 or vol_conf or panic_res) else "MEDIUM"
                thesis = f"Value pullback bounce off dynamic support.{vol_tag}{extra_insight} Favorable asymmetric R:R."
                trailing_playbook = f"Book 33% profit at T1 (₹{target_1:,.1f}) & trail SL to Entry (₹{entry:,.1f}) for a risk-free trade. Book next 33% at T2 (₹{target_2:,.1f}) & trail SL to T1. Let remaining 34% runners target T3 & T4."

            elif cat == "INTRADAY":
                bias = c.get("bias", "LONG")
                if bias == "LONG":
                    entry = close
                    tight_sl = round(float(metrics.get("day_low", close - (0.6 * atr))), 2)
                    cons_sl = round(close - (1.0 * atr), 2)
                    stop_loss = tight_sl
                    risk = max(entry - stop_loss, close * 0.005)
                    target_1 = round(entry + (1.0 * risk), 2)
                    target_2 = round(entry + (1.8 * risk), 2)
                    target_3 = round(entry + (2.5 * risk), 2)
                    target_4 = round(entry + (3.5 * risk), 2)
                    action = "BUY"
                else:
                    entry = close
                    tight_sl = round(float(metrics.get("day_high", close + (0.6 * atr))), 2)
                    cons_sl = round(close + (1.0 * atr), 2)
                    stop_loss = tight_sl
                    risk = max(stop_loss - entry, close * 0.005)
                    target_1 = round(entry - (1.0 * risk), 2)
                    target_2 = round(entry - (1.8 * risk), 2)
                    target_3 = round(entry - (2.5 * risk), 2)
                    target_4 = round(entry - (3.5 * risk), 2)
                    action = "SELL"
                timeframe = "Intraday (Same Day Exit)"
                conviction = "HIGH" if (c.get("score", 0) >= 80 or vol_conf) else "MEDIUM"
                thesis = f"Intraday momentum expansion ({vol_mult:.1f}x volume) with strong directional CLV.{vol_tag}{extra_insight}"
                trailing_playbook = f"Intraday setup: Book 50% at T1 (₹{target_1:,.1f}) and trail SL to cost. Exit 100% before 3:15 PM IST."

            elif cat == "OPTIONS":
                opt_type = c.get("option_type", "CE")
                strike = c.get("recommended_strike", close)
                action = f"BUY_{opt_type}"
                entry = close
                tight_sl = round(close - (1.0 * atr) if opt_type == "CE" else close + (1.0 * atr), 2)
                cons_sl = round(close - (1.5 * atr) if opt_type == "CE" else close + (1.5 * atr), 2)
                stop_loss = tight_sl
                risk = max(abs(entry - stop_loss), close * 0.015)
                target_1 = round(close + (1.2 * atr) if opt_type == "CE" else close - (1.2 * atr), 2)
                target_2 = round(close + (2.0 * atr) if opt_type == "CE" else close - (2.0 * atr), 2)
                target_3 = round(close + (3.0 * atr) if opt_type == "CE" else close - (3.0 * atr), 2)
                target_4 = round(close + (4.2 * atr) if opt_type == "CE" else close - (4.2 * atr), 2)
                timeframe = "Weekly / Monthly Expiry"
                conviction = "HIGH" if (c.get("score", 0) >= 80 or vol_conf) else "MEDIUM"
                thesis = f"Directional surge triggering {strike:.0f} {opt_type} option play with favorable delta.{vol_tag}{extra_insight}"
                trailing_playbook = f"Option Delta Surge: Book 50% profit at T1 (₹{target_1:,.1f}) and trail SL to option premium breakeven."

            elif cat in ["DIP_LEADER", "DIP"]:
                entry = close
                invalidation_sl = float(c.get("invalidation_sl", round(close * 0.97, 2)))
                tight_sl = invalidation_sl
                cons_sl = round(invalidation_sl * 0.985, 2)
                stop_loss = tight_sl
                risk = max(entry - stop_loss, close * 0.015)
                target_1 = round(float(c.get("rebound_target_1", entry + (1.6 * risk))), 2)
                target_2 = round(float(c.get("rebound_target_2", entry + (2.8 * risk))), 2)
                target_3 = round(float(c.get("rebound_target_3", entry + (4.0 * risk))), 2)
                target_4 = round(entry + (5.0 * risk), 2)
                action = "BUY"
                timeframe = "1-3 weeks (Market Dip Springboard)"
                conviction = "HIGH" if (c.get("score", 0) >= 80 or c.get("green_on_dips", 0) >= 1) else "MEDIUM"
                rec_badge = c.get("recovery_velocity_badge", "⚡ ULTRA-FAST BOUNCE")
                alpha_val = c.get("dip_alpha", 0.0)
                thesis = (
                    f"{rec_badge}: Superior relative strength on market dips (Alpha: {alpha_val:+.1f}%) "
                    f"with dry selling volume ({vol_mult:.1f}x). Prime coiled spring to lead market recovery as Nifty stabilizes.{extra_insight}"
                )
                trailing_playbook = (
                    f"Accumulate in springboard zone (₹{entry:,.1f}). Book 40% at T1 (₹{target_1:,.1f}) "
                    f"and trail SL immediately to Entry for a risk-free trade. Book next 30% at T2 (₹{target_2:,.1f}) "
                    f"and trail SL to T1. Let remaining 30% runners target T3/T4 (₹{target_3:,.1f})."
                )

            elif cat == "TWO_WAY_LEVELS":
                bias = c.get("bias", "LONG")
                rise_above = float(c.get("rise_above", close * 1.015))
                fall_below = float(c.get("fall_below", close * 0.985))
                up_t1 = float(c.get("upside_target_1", rise_above + (1.2 * atr)))
                down_t1 = float(c.get("downside_target_1", fall_below - (1.2 * atr)))
                upside_sl = float(c.get("upside_stop_loss", rise_above - (0.5 * atr)))

                # Check for upside SL breach:
                # If current price is below the upside SL, the bullish thesis is invalidated.
                upside_invalidated = bool(close < upside_sl or c.get("upside_sl_breached", False))
                is_short = (bias == "SHORT" or "BEAR" in str(bias).upper() or upside_invalidated or close <= fall_below)

                if is_short:
                    entry = fall_below
                    tight_sl = round(fall_below + (0.6 * atr), 2)
                    cons_sl = round(fall_below + (1.2 * atr), 2)
                    stop_loss = tight_sl
                    risk = max(stop_loss - entry, close * 0.01)
                    target_1 = down_t1
                    target_2 = round(fall_below - (1.6 * atr), 2)
                    target_3 = round(fall_below - (2.5 * atr), 2)
                    target_4 = round(fall_below - (3.5 * atr), 2)
                    action = "SELL"
                    timeframe = "Breakdown Floor Trigger"
                    conviction = "HIGH" if (c.get("score", 0) >= 80 or close <= fall_below) else "MEDIUM"
                    if upside_invalidated:
                        thesis = (
                            f"🛑 Bullish SL Breached: Stock dropped to ₹{close:,.2f}, breaching prior upside support (₹{upside_sl:,.2f}). "
                            f"Bullish setup is invalidated. Active tactical trade is SHORT on breakdown below floor ₹{fall_below:,.2f} toward ₹{target_1:,.2f}."
                        )
                        trailing_playbook = (
                            f"Do NOT buy at market. Wait for breakdown below ₹{fall_below:,.2f} (SL: ₹{stop_loss:,.2f}). "
                            f"Cover 50% at T1 (₹{target_1:,.2f}) and trail SL to cost."
                        )
                    else:
                        thesis = f"Two-Way Trigger: Breakdown below floor ₹{fall_below:,.2f} invalidates demand toward ₹{target_1:,.2f}."
                        trailing_playbook = f"Sell short on decisive breakdown below ₹{fall_below:,.2f} (SL: ₹{stop_loss:,.2f}). Cover 50% at T1 (₹{target_1:,.2f})."
                else:
                    entry = rise_above
                    tight_sl = round(rise_above - (0.6 * atr), 2)
                    cons_sl = round(rise_above - (1.2 * atr), 2)
                    stop_loss = tight_sl
                    risk = max(entry - stop_loss, close * 0.01)
                    target_1 = up_t1
                    target_2 = round(rise_above + (1.6 * atr), 2)
                    target_3 = round(rise_above + (2.5 * atr), 2)
                    target_4 = round(rise_above + (3.5 * atr), 2)
                    action = "BUY"
                    timeframe = "Breakout Ceiling Trigger"
                    conviction = "HIGH" if c.get("score", 0) >= 80 else "MEDIUM"
                    thesis = f"Two-Way Trigger: Rally break above resistance ₹{rise_above:,.2f} triggers upside expansion toward ₹{target_1:,.2f}."
                    trailing_playbook = f"Buy on sustained breakout above ₹{rise_above:,.2f} (SL: ₹{stop_loss:,.2f}). Book 50% at T1 (₹{target_1:,.2f})."

            else:
                entry = close
                tight_sl = round(close * 0.98, 2)
                cons_sl = round(close * 0.96, 2)
                stop_loss = tight_sl
                risk = max(entry - stop_loss, close * 0.01)
                target_1 = round(close * 1.02, 2)
                target_2 = round(close * 1.04, 2)
                target_3 = round(close * 1.07, 2)
                target_4 = round(close * 1.10, 2)
                action = "BUY"
                timeframe = "Swing"
                conviction = "MEDIUM"
                thesis = "Technical setup meeting initial volume and momentum criteria."
                trailing_playbook = "Standard swing: Book 50% at T1 and trail SL to cost."

            # Evaluate macroeconomic forces & sector sensitivities (Crude Oil, Currency, VIX)
            sym = c.get("symbol", "")
            macro_impact = MacroMarketEngine.get_stock_macro_impact(sym, macro_context)
            macro_badge = macro_impact.get("badge", "")
            macro_thesis = macro_impact.get("thesis", "")
            
            if macro_impact.get("has_headwind"):
                thesis = f"{thesis} | {macro_badge}: {macro_thesis}"
                if conviction == "HIGH" and macro_impact.get("conviction_delta", 0) <= -12:
                    conviction = "MEDIUM"
            elif macro_impact.get("has_tailwind"):
                thesis = f"{thesis} | {macro_badge}: {macro_thesis}"
                if conviction == "MEDIUM" and macro_impact.get("conviction_delta", 0) >= 12:
                    conviction = "HIGH"

            # Raw plan dictionary
            risk = abs(entry - stop_loss)
            reward_t1 = abs(target_1 - entry)
            reward_t4 = abs(target_4 - entry)
            rr_ratio = f"1:{reward_t1 / (risk + 1e-6):.1f} (T1) → 1:{reward_t4 / (risk + 1e-6):.1f} (T4)"
            shares_for_2k_risk = max(int(2000 / (risk + 1e-6)), 1)
            capital_for_2k_risk = round(shares_for_2k_risk * entry, 2)

            raw_plan = {
                "symbol": c.get("symbol"),
                "category": cat,
                "trade_action": action,
                "close": close,
                "change_pct": float(c.get("change_pct", 0.0)),
                "entry_price": entry,
                "stop_loss": stop_loss,
                "tight_stop_loss": tight_sl,
                "conservative_stop_loss": cons_sl,
                "target_1": target_1,
                "target_2": target_2,
                "target_3": target_3,
                "target_4": target_4,
                "risk_reward_ratio": rr_ratio,
                "timeframe": timeframe,
                "conviction": conviction,
                "thesis": thesis,
                "trailing_playbook": trailing_playbook,
                "shares_for_2k_risk": shares_for_2k_risk,
                "capital_for_2k_risk": capital_for_2k_risk,
                "macro_badge": macro_badge,
                "score": float(c.get("score", 75.0)),
                "rvol": float(c.get("rvol", 1.0)),
                "rvol_9": float(c.get("rvol_9", c.get("rvol", 1.0))),
                "vol_confirmed_2x": bool(vol_conf),
                "retest_support_50pct": float(retest_lvl),
                "has_sector_tailwind": bool(has_tailwind),
                "panic_day_resilient": bool(panic_res),
                "green_on_dips": int(c.get("green_on_dips", 0)),
                "sector": c.get("sector", "")
            }
            # Guard against SL inversion or invalidation
            guarded_plan = self._sanitize_and_guard_plan(raw_plan, c)
            plans.append(guarded_plan)

        return plans

    @staticmethod
    def _sanitize_and_guard_plan(plan: Dict[str, Any], cand: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Guarantees strict mathematical progression and prevents Stop Loss inversion against CMP.
        If current market price is at or below the stop loss for a BUY trade, the setup is invalidated
        and either flipped to a breakdown SHORT plan (for TWO_WAY_LEVELS) or marked as AVOID.
        """
        if cand:
            for k in ["score", "rvol", "rvol_9", "vol_confirmed_2x", "retest_support_50pct", 
                      "has_sector_tailwind", "panic_day_resilient", "green_on_dips", "sector"]:
                if k not in plan and k in cand:
                    plan[k] = cand[k]
        close = float(plan.get("close", 0.0))
        if close <= 0 and cand:
            close = float(cand.get("close", 0.0))
            plan["close"] = close

        entry = float(plan.get("entry_price", close))
        stop_loss = float(plan.get("stop_loss", close * 0.95))
        tight_sl = float(plan.get("tight_stop_loss", stop_loss))
        cons_sl = float(plan.get("conservative_stop_loss", stop_loss))
        target_1 = float(plan.get("target_1", close * 1.02))
        target_2 = float(plan.get("target_2", close * 1.04))
        target_3 = float(plan.get("target_3", close * 1.06))
        target_4 = float(plan.get("target_4", close * 1.08))

        action = str(plan.get("trade_action", "BUY")).upper()
        cat = plan.get("category", "")
        atr = float(cand.get("atr", close * 0.02)) if cand else (close * 0.02)
        if atr <= 0:
            atr = max(close * 0.02, 0.5)

        is_bullish = ("BUY" in action or action == "LONG") and "PE" not in action

        # 1. Anti-Inversion & Invalidation Check against Live Market Price (close / CMP)
        if is_bullish and close > 0 and close <= stop_loss:
            if cat == "TWO_WAY_LEVELS":
                plan["trade_action"] = "SELL"
                is_bullish = False
                f_below = float(cand.get("fall_below", round(close * 0.99, 2))) if cand else round(close * 0.99, 2)
                entry = f_below
                stop_loss = round(max(entry + (0.8 * atr), close + (0.5 * atr)), 2)
                tight_sl = stop_loss
                cons_sl = round(max(entry + (1.2 * atr), tight_sl + (0.3 * atr)), 2)
                target_1 = float(cand.get("downside_target_1", round(entry - (1.2 * atr), 2))) if cand else round(entry - (1.2 * atr), 2)
                target_2 = round(entry - (1.6 * atr), 2)
                target_3 = round(entry - (2.5 * atr), 2)
                target_4 = round(entry - (3.5 * atr), 2)
                plan["timeframe"] = "Breakdown Floor Trigger"
                plan["conviction"] = "HIGH" if (cand and cand.get("score", 0) >= 80) else "MEDIUM"
                plan["thesis"] = (
                    f"🛑 Bullish SL Breached: Stock fell to ₹{close:,.2f}, breaching prior support/SL. "
                    f"Bullish setup is void. Tactical trade is SHORT on breakdown below floor ₹{entry:,.2f} toward ₹{target_1:,.2f}."
                )
                plan["trailing_playbook"] = f"Do NOT buy at market. Sell on breakdown below ₹{entry:,.2f} (SL: ₹{stop_loss:,.2f}). Cover 50% at T1 (₹{target_1:,.2f})."
            else:
                plan["trade_action"] = "AVOID"
                plan["timeframe"] = "Setup Invalidated (SL Broken)"
                plan["thesis"] = (
                    f"🛑 SETUP INVALIDATED: Stock traded down to ₹{close:,.2f}, breaching critical stop loss (₹{stop_loss:,.2f}). "
                    f"Bullish thesis is void at market price."
                )
                plan["trailing_playbook"] = "Avoid entering or square off existing position immediately. Wait for fresh institutional base."
                plan["conviction"] = "SPECULATIVE"
                is_bullish = False

        # 2. Strict Progression and Distance Enforcement
        risk = max(abs(entry - stop_loss), close * 0.01)
        if is_bullish:
            # Bullish trades: SL must be strictly below BOTH entry and close (for market entry)
            if abs(entry - close) / max(close, 1.0) < 0.02:
                stop_loss = min(stop_loss, round(close - (0.5 * risk), 2))
            else:
                stop_loss = min(stop_loss, round(min(entry, close - 0.1), 2))
            tight_sl = min(tight_sl, stop_loss)
            cons_sl = min(cons_sl, tight_sl)
            target_1 = max(target_1, round(entry + (1.0 * risk), 2))
            target_2 = max(target_2, round(target_1 + (0.5 * risk), 2))
            target_3 = max(target_3, round(target_2 + (0.6 * risk), 2))
            target_4 = max(target_4, round(target_3 + (0.8 * risk), 2))
        elif plan["trade_action"] in ["SELL", "SHORT"]:
            # Bearish trades: SL must be strictly above BOTH entry and close
            stop_loss = max(stop_loss, round(max(entry, close) + (0.5 * risk), 2))
            tight_sl = max(tight_sl, stop_loss)
            cons_sl = max(cons_sl, tight_sl)
            target_1 = min(target_1, round(entry - (1.0 * risk), 2))
            target_2 = min(target_2, round(target_1 - (0.5 * risk), 2))
            target_3 = min(target_3, round(target_2 - (0.6 * risk), 2))
            target_4 = min(target_4, round(target_3 - (0.8 * risk), 2))

        risk = max(abs(entry - stop_loss), 0.05)
        reward_t1 = abs(target_1 - entry)
        reward_t4 = abs(target_4 - entry)
        plan["entry_price"] = entry
        plan["stop_loss"] = stop_loss
        plan["tight_stop_loss"] = tight_sl
        plan["conservative_stop_loss"] = cons_sl
        plan["target_1"] = target_1
        plan["target_2"] = target_2
        plan["target_3"] = target_3
        plan["target_4"] = target_4
        plan["risk_reward_ratio"] = f"1:{reward_t1 / risk:.1f} (T1) → 1:{reward_t4 / risk:.1f} (T4)"
        plan["shares_for_2k_risk"] = max(int(2000 / risk), 1)
        plan["capital_for_2k_risk"] = round(plan["shares_for_2k_risk"] * entry, 2)

        # Automatic institutional confirmation tagging
        is_conf, conf_factors, conf_score = AIAdvisor.evaluate_plan_confirmation(plan, cand)
        plan["is_confirmed"] = is_conf
        plan["confirmation_factors"] = conf_factors
        plan["safety_score"] = conf_score
        return plan

    @staticmethod
    def evaluate_plan_confirmation(plan: Dict[str, Any], cand: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], int]:
        """
        Determines whether a trade plan meets strict institutional 'Confirmed Strategy' standards
        that reliably perform without failing under normal market situations.
        """
        action = str(plan.get("trade_action", "")).upper()
        if "AVOID" in action:
            return False, [], 0
        
        # Conviction must be HIGH
        conv_str = str(plan.get("conviction", "")).upper()
        if "HIGH" not in conv_str:
            return False, [], 0
            
        # No Macro Headwinds
        macro_b = str(plan.get("macro_badge", "")).upper()
        if "HEADWIND" in macro_b or "BEARISH DRAG" in macro_b:
            return False, [], 0
            
        # Check current price vs stop loss
        try:
            entry = float(plan.get("entry_price", 0.0))
            sl = float(plan.get("tight_stop_loss", plan.get("stop_loss", 0.0)))
            cmp = float(plan.get("close", entry))
            is_bullish = ("BUY" in action or action == "LONG") and "PE" not in action
            if is_bullish and cmp <= sl and cmp > 0 and sl > 0:
                return False, [], 0
            if not is_bullish and cmp >= sl and cmp > 0 and sl > 0:
                return False, [], 0
        except Exception:
            pass
            
        # Risk:Reward to Target 1 must be >= 1.0
        try:
            entry = float(plan.get("entry_price", 0.0))
            sl = float(plan.get("stop_loss", 0.0))
            t1 = float(plan.get("target_1", 0.0))
            risk = abs(entry - sl)
            if risk > 0 and (abs(t1 - entry) / risk) < 1.0:
                return False, [], 0
        except Exception:
            pass

        factors = []
        rvol_val = float(plan.get("rvol_9", plan.get("rvol", cand.get("rvol_9", cand.get("rvol", 1.0)) if cand else 1.0)))
        vol_2x = bool(plan.get("vol_confirmed_2x", cand.get("vol_confirmed_2x", False) if cand else False))
        if vol_2x or rvol_val >= 1.5:
            factors.append(f"🔥 Vol Surge ({rvol_val:.1f}x)")
        elif rvol_val >= 1.2:
            factors.append(f"📊 Vol Expansion ({rvol_val:.1f}x)")

        sc = float(plan.get("score", cand.get("score", 0.0) if cand else 0.0))
        if sc >= 80:
            factors.append(f"⭐ Quant Score {sc:.0f}/100")
        elif sc >= 75:
            factors.append(f"📈 Momentum {sc:.0f}/100")

        macro_badge_raw = str(plan.get("macro_badge", "")).strip()
        has_tailwind = bool(plan.get("has_sector_tailwind", cand.get("has_sector_tailwind", False) if cand else False))
        sector_name = plan.get("sector", cand.get("sector", "Sector") if cand else "Sector")
        if macro_badge_raw and "HEADWIND" not in macro_badge_raw.upper():
            if "TAILWIND" in macro_badge_raw.upper() or "BENEFICIARY" in macro_badge_raw.upper():
                factors.append(f"🌊 {macro_badge_raw}")
            else:
                factors.append(f"🌍 {macro_badge_raw}")
        elif has_tailwind:
            factors.append(f"🌊 {sector_name} Tailwind")

        retest = float(plan.get("retest_support_50pct", cand.get("retest_support_50pct", 0.0) if cand else 0.0))
        if retest > 0:
            factors.append("🛡️ Retest Defense Held")

        if plan.get("panic_day_resilient", cand.get("panic_day_resilient", False) if cand else False):
            factors.append("💪 Panic-Day Resilient")
        if plan.get("green_on_dips", cand.get("green_on_dips", 0) if cand else 0) >= 1 or plan.get("category") in ["DIP_LEADER", "DIP"]:
            factors.append("🚀 Dip Recovery Leader")

        try:
            entry = float(plan.get("entry_price", 0.0))
            sl = float(plan.get("stop_loss", 0.0))
            t1 = float(plan.get("target_1", 0.0))
            risk = abs(entry - sl)
            if risk > 0 and (abs(t1 - entry) / risk) >= 1.4:
                factors.append(f"🎯 Favorable 1:{(abs(t1 - entry) / risk):.1f} R:R")
        except Exception:
            pass

        if not factors:
            factors.append("✅ Algorithmic Momentum Aligned")

        safety_score = min(85 + (len(factors) * 3), 98)
        return True, factors, safety_score
