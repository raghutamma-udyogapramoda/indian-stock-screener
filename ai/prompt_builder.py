"""
Prompt Builder for LLM Trade Advisor.
Transforms shortlisted candidates into an ultra-compact JSON feature vector.
Shields against token bloat and keeps LLM cost below $0.001 per screening run.
"""

import json
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """Builds efficient, structured prompts for the LLM."""

    SYSTEM_PROMPT = """You are an elite Indian stock market technical strategist, macro economist, and quantitative trading mentor.
Analyze the provided high-conviction trade setups screened from NSE/BSE.

CRITICAL MACRO INSTRUCTION:
You MUST actively evaluate the broader market regime and cross-asset macroeconomic drivers:
1. Broader Market & VIX: If NIFTY is correcting or volatile (VIX > 16), demand higher relative strength and tighten stop losses.
2. Crude Oil Impact:
   - High / Rising Crude (> $82/bbl) is a major HEADWIND for Paints, Tyres, Aviation, and OMCs (margin compression). Downgrade conviction on long breakouts in these sectors.
   - Surging Crude is a major TAILWIND for Upstream Oil & Gas (ONGC, OIL).
3. Currency (USD/INR): Elevated USD/INR provides structural export tailwinds for IT Software (TCS, INFY) and Pharma.
4. First-to-Recover Dip Leaders (Category: DIP_LEADER):
   - When the market dips, institutions accumulate elite Stage-2 leaders. These stocks display dry selling volume and strong relative strength on red days, making them the FIRST and FASTEST to rocket upward when Nifty rebounds. Prioritize these on market pullbacks.
5. Two-Way Action Levels (Category: TWO_WAY_LEVELS):
   - If the bias is SHORT or BEARISH, set trade_action to SELL below the downside floor trigger.
   - If the bias is LONG or BULLISH, set trade_action to BUY above the upside breakout trigger.

For each stock, evaluate both quantitative metrics and macro alignment, and output a strict JSON array of objects with the following schema:
[
  {
    "symbol": "TICKER",
    "category": "BREAKOUT | BREAKDOWN | SWING | BTST | INTRADAY | OPTIONS | DIP_LEADER | TWO_WAY_LEVELS",
    "trade_action": "BUY | SELL | BUY_CE | BUY_PE | AVOID",
    "entry_price": 0.0,
    "stop_loss": 0.0,
    "tight_stop_loss": 0.0,
    "conservative_stop_loss": 0.0,
    "target_1": 0.0,
    "target_2": 0.0,
    "target_3": 0.0,
    "target_4": 0.0,
    "risk_reward_ratio": "1:1.5 → 1:3.5",
    "timeframe": "1-2 days | Intraday | 1-4 weeks",
    "conviction": "HIGH | MEDIUM | SPECULATIVE",
    "thesis": "Concise 2-sentence rationale highlighting key triggers, macro tailwinds/headwinds, and recovery dynamics.",
    "trailing_playbook": "Actionable trailing plan (e.g. Book 33% at T1 & trail SL to Cost. Book 33% at T2 & trail SL to T1. Hold runners for T3 & T4).",
    "shares_for_2k_risk": 0
  }
]
Return ONLY valid JSON without markdown wrapping or explanations.
"""

    @staticmethod
    def build_user_prompt(candidates: List[Dict[str, Any]], macro_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Serializes quantitative candidate metrics and macroeconomic environment into clean, compact JSON for the LLM.
        """
        sanitized_macro = {}
        if macro_context:
            sanitized_macro = {
                "regime_badge": macro_context.get("regime_badge"),
                "nifty_trend": f"NIFTY: {macro_context.get('nifty', {}).get('close')} ({macro_context.get('nifty', {}).get('change_pct'):+.2f}%, {macro_context.get('nifty', {}).get('trend')})",
                "vix_volatility": f"INDIA VIX: {macro_context.get('vix', {}).get('level')} ({macro_context.get('vix', {}).get('risk_tier')})",
                "crude_oil": f"Crude Oil: ${macro_context.get('crude', {}).get('usd_price')}/bbl ({macro_context.get('crude', {}).get('regime')})",
                "usdinr_fx": f"USD/INR: {macro_context.get('usdinr', {}).get('rate')} ({macro_context.get('usdinr', {}).get('regime')})",
                "tactical_guidance": macro_context.get("regime_desc")
            }

        sanitized = []
        for c in candidates:
            sanitized.append({
                "symbol": c.get("symbol"),
                "category": c.get("category"),
                "ltp": c.get("close"),
                "change_pct": c.get("change_pct"),
                "rvol_9d": c.get("rvol_9", c.get("rvol")),
                "vol_confirmed_2x": c.get("vol_confirmed_2x", False),
                "retest_support_50pct": c.get("retest_support_50pct", 0.0),
                "retracement_healthy": c.get("retracement_healthy", True),
                "sector": c.get("sector", "Broad Market"),
                "has_sector_tailwind": c.get("has_sector_tailwind", False),
                "panic_day_resilient": c.get("panic_day_resilient", False),
                "mansfield_rs": c.get("mansfield_rs", 0.0),
                "dip_alpha": c.get("dip_alpha", 0.0),
                "upside_beta": c.get("upside_beta", 1.0),
                "recovery_badge": c.get("recovery_velocity_badge", ""),
                "macro_badge": c.get("macro_badge", ""),
                "rsi_14": c.get("rsi"),
                "triggers": c.get("reasons", []),
                "key_levels": c.get("metrics", {})
            })

        payload = {
            "macro_environment": sanitized_macro,
            "candidates_to_evaluate": sanitized
        }
        return f"Market Macro Context & Trade Candidates:\n{json.dumps(payload, indent=2)}"
