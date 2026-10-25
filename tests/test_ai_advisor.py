"""
Unit tests for AIAdvisor initialization, default Gemini key loading, and user override capabilities.
"""
import unittest
from config.settings import DEFAULT_GEMINI_API_KEY, GEMINI_API_KEY
from ai.advisor import AIAdvisor


class TestAIAdvisorConfigAndOverride(unittest.TestCase):
    def test_default_gemini_api_key_loaded(self):
        """Verifies that default Gemini API key is configured and AIAdvisor uses it by default."""
        self.assertTrue(len(DEFAULT_GEMINI_API_KEY) > 20 and DEFAULT_GEMINI_API_KEY.startswith("AQ."))
        advisor = AIAdvisor()
        self.assertEqual(advisor.api_key, DEFAULT_GEMINI_API_KEY)
        self.assertTrue(advisor.is_ai_ready)

    def test_user_override_gemini_api_key(self):
        """Verifies that user can supply a custom Gemini API key to override default."""
        custom_key = "custom_user_override_key_98765"
        advisor = AIAdvisor(api_key=custom_key)
        self.assertEqual(advisor.api_key, custom_key)

    def test_user_clear_gemini_api_key_fallback(self):
        """Verifies that clearing the API key explicitly disables live LLM and falls back to mathematical precision engine."""
        advisor = AIAdvisor(api_key="")
        self.assertEqual(advisor.api_key, "")
        self.assertFalse(advisor.is_ai_ready)
        self.assertIsNone(advisor.client)

    def test_analyze_candidates_offline_fallback_works(self):
        """Verifies candidate analysis works smoothly even if API key is cleared (mathematical engine)."""
        advisor = AIAdvisor(api_key="")
        sample_candidates = [
            {
                "symbol": "RELIANCE",
                "category": "Breakout",
                "bias": "LONG",
                "close": 2500.0,
                "change_pct": 2.0,
                "rvol": 1.8,
                "reasons": ["Supertrend Breakout"],
                "metrics": {"breakout_pivot": 2450.0, "atr": 35.0},
            }
        ]
        plans = advisor.analyze_candidates(sample_candidates)
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["symbol"], "RELIANCE")
        self.assertIn("entry_price", plans[0])
        self.assertIn("stop_loss", plans[0])
        self.assertIn("target_1", plans[0])


if __name__ == "__main__":
    unittest.main()
