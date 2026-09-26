"""
Prompt Builder for LLM Trade Advisor.
Transforms shortlisted candidates into an ultra-compact JSON feature vector.
Shields against token bloat and keeps LLM cost below $0.001 per screening run.
"""

import json
from typing import Any, Dict, List


class PromptBuilder:
    """Builds efficient, structured prompts for the LLM."""

    SYSTEM_PROMPT = """You are an elite Indian stock market technical strategist and quantitative trading mentor.
Analyze the provided high-conviction trade setups screened from NSE/BSE.

For each stock, evaluate the quantitative indicators and output a strict JSON array of objects with the following schema:
[
  {
    "symbol": "TICKER",
    "category": "BREAKOUT | BREAKDOWN | SWING | BTST | INTRADAY | OPTIONS",
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
    "thesis": "Concise 2-sentence rationale highlighting key triggers, risk factors, and market rationale.",
    "trailing_playbook": "Actionable trailing plan (e.g. Book 33% at T1 & trail SL to Cost. Book 33% at T2 & trail SL to T1. Hold runners for T3 & T4).",
    "shares_for_2k_risk": 0
  }
]
Return ONLY valid JSON without markdown wrapping or explanations.
"""

    @staticmethod
    def build_user_prompt(candidates: List[Dict[str, Any]]) -> str:
        """
        Serializes quantitative candidate metrics into clean, compact JSON for the LLM.
        
        Token Shield Mechanism:
          - Eliminates all raw OHLCV time-series rows (avoids sending 90,000+ data points).
          - Condenses each qualified stock down to 8 essential decision metrics:
              1. 'symbol': NSE ticker (e.g. 'RELIANCE', 'SUNTV')
              2. 'category': Specific trade strategy (BREAKOUT, BREAKDOWN, SWING, BTST, INTRADAY, OPTIONS)
              3. 'ltp': Last Traded Price in ₹
              4. 'change_pct': Session percentage change
              5. 'rvol': 20-day Relative Volume surge multiplier
              6. 'rsi_14': Wilder's 14-period RSI
              7. 'triggers': List of human-readable quantitative triggers that fired
              8. 'key_levels': Critical resistance pivots, support levels, dynamic EMAs, and option strikes
          - Total payload size: < 1,500 tokens (~$0.0004 cost on Gemini Flash).
        """
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
                "rsi_14": c.get("rsi"),
                "triggers": c.get("reasons", []),
                "key_levels": c.get("metrics", {})
            })

        return f"Candidates for trade evaluation:\n{json.dumps(sanitized, indent=2)}"
