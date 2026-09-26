"""
AI Trade Advisor powered by Google Gemini.
Evaluates shortlisted quantitative setups and generates actionable trade plans.
"""

import json
import os
from typing import Any, Dict, List, Optional

from ai.prompt_builder import PromptBuilder
from config.settings import GEMINI_API_KEY, GEMINI_MODEL


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

    def analyze_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Passes shortlisted setups to Gemini.
        If API key is missing or calls fail, generates an algorithmic trade plan so workflow is uninterrupted.
        """
        if not candidates:
            return []

        # If live AI is available, query Gemini
        if self.is_ai_ready:
            try:
                user_content = PromptBuilder.build_user_prompt(candidates)
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
                        return parsed
            except Exception:
                # Smooth fallback to algorithmic precision engine
                pass

        # Algorithmic fallback: generates mathematically sound trade plan
        return self._generate_algorithmic_trade_plan(candidates)

    def _generate_algorithmic_trade_plan(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates mathematically precise Entry, Stop-Loss, and Target levels.
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

            # Strict Mathematical Progression Guard: Guarantee targets never overlap or invert
            is_bullish = ("BUY" in action or action == "LONG") and "PE" not in action
            risk = max(abs(entry - stop_loss), close * 0.01)

            if is_bullish:
                # Bullish trade: SL < Entry < T1 < T2 < T3 < T4
                stop_loss = min(stop_loss, round(entry - (0.8 * risk), 2))
                tight_sl = min(tight_sl, round(entry - (0.5 * risk), 2))
                cons_sl = min(cons_sl, tight_sl)
                target_1 = max(target_1, round(entry + (1.0 * risk), 2))
                target_2 = max(target_2, round(target_1 + (0.5 * risk), 2))
                target_3 = max(target_3, round(target_2 + (0.6 * risk), 2))
                target_4 = max(target_4, round(target_3 + (0.8 * risk), 2))
            else:
                # Bearish / Short trade: SL > Entry > T1 > T2 > T3 > T4
                stop_loss = max(stop_loss, round(entry + (0.8 * risk), 2))
                tight_sl = max(tight_sl, round(entry + (0.5 * risk), 2))
                cons_sl = max(cons_sl, tight_sl)
                target_1 = min(target_1, round(entry - (1.0 * risk), 2))
                target_2 = min(target_2, round(target_1 - (0.5 * risk), 2))
                target_3 = min(target_3, round(target_2 - (0.6 * risk), 2))
                target_4 = min(target_4, round(target_3 - (0.8 * risk), 2))

            risk = abs(entry - stop_loss)
            reward_t1 = abs(target_1 - entry)
            reward_t4 = abs(target_4 - entry)
            rr_ratio = f"1:{reward_t1 / (risk + 1e-6):.1f} (T1) → 1:{reward_t4 / (risk + 1e-6):.1f} (T4)"
            
            # Position sizing rule: Units to buy for fixed ₹2,000 risk
            shares_for_2k_risk = max(int(2000 / (risk + 1e-6)), 1)
            capital_for_2k_risk = round(shares_for_2k_risk * entry, 2)

            plans.append({
                "symbol": c.get("symbol"),
                "category": cat,
                "trade_action": action,
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
                "capital_for_2k_risk": capital_for_2k_risk
            })

        return plans
