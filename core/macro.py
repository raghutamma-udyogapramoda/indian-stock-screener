"""
Broader Market & Macroeconomic Intelligence Engine for Indian Equities.
Analyzes macroeconomic drivers (NIFTY 50, BANK NIFTY, INDIA VIX, Crude Oil, USD/INR, Gold)
and evaluates their direct structural impact on specific stocks and sectors.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.universe import UniverseManager


# Comprehensive Mapping of Macro-Sensitive Indian Equities
CRUDE_SENSITIVE_CONSUMERS = {
    # Paints & Coatings (Raw materials are 50%+ crude derivatives like titanium dioxide, monomers)
    "ASIANPAINT": "Paints & Coatings",
    "BERGEPAINT": "Paints & Coatings",
    "KANSAINER": "Paints & Coatings",
    "AKZOINDIA": "Paints & Coatings",
    "PIDILITIND": "Adhesives & Chemicals",
    
    # Tyres & Rubber (Synthetic rubber & carbon black are petrochemical derivatives)
    "APOLLOTYRE": "Tyres & Rubber",
    "MRF": "Tyres & Rubber",
    "BALKRISIND": "Tyres & Rubber",
    "CEATLTD": "Tyres & Rubber",
    "JKTYRE": "Tyres & Rubber",
    
    # Aviation (Aviation Turbine Fuel / ATF is 40%+ of operating expenses)
    "INDIGO": "Aviation",
    "SPICEJET": "Aviation",
    
    # Oil Marketing Companies (Retail fuel price caps squeeze marketing margins on high crude)
    "BPCL": "Oil Marketing",
    "HPCL": "Oil Marketing",
    "IOC": "Oil Marketing",
    
    # Specialty Chemicals (Petrochemical feedstock costs)
    "AARTIIND": "Specialty Chemicals",
    "DEEPAKNTR": "Specialty Chemicals",
    "SRF": "Specialty Chemicals",
    "NAVINFLUOR": "Specialty Chemicals",
    "ATUL": "Specialty Chemicals",
    
    # Automobile (Input cost inflation & consumer sentiment)
    "MARUTI": "Automobile",
    "TMPV": "Automobile",
    "TATAMOTORS": "Automobile",
    "M&M": "Automobile",
    "HEROMOTOCO": "Two-Wheelers",
    "BAJAJ-AUTO": "Two-Wheelers",
    "TVSMOTOR": "Two-Wheelers",
}

CRUDE_UPSTREAM_BENEFICIARIES = {
    # Upstream Exploration & Production (Direct realization expansion on rising crude)
    "ONGC": "Upstream Oil & Gas",
    "OIL": "Upstream Oil & Gas",
    "GAIL": "Gas Transmission",
    "RELIANCE": "Diversified Refining & Energy",
}

USDINR_EXPORT_BENEFICIARIES = {
    # IT & Software Services (70-85% revenue in USD/EUR; Rupee depreciation directly expands EBIT margins)
    "TCS": "IT Exporter",
    "INFY": "IT Exporter",
    "HCLTECH": "IT Exporter",
    "WIPRO": "IT Exporter",
    "TECHM": "IT Exporter",
    "LTIM": "IT Exporter",
    "PERSISTENT": "IT Exporter",
    "COFORGE": "IT Exporter",
    "MPHASIS": "IT Exporter",
    "LTTS": "IT Exporter",
    "TATAELXSI": "IT Exporter",
    "BSOFT": "IT Exporter",
    "OFSS": "IT Exporter",
    "NAUKRI": "Internet & IT",

    # Pharmaceuticals & Generics Exporters (Substantial US/Europe revenue)
    "SUNPHARMA": "Pharma Exporter",
    "DRREDDY": "Pharma Exporter",
    "CIPLA": "Pharma Exporter",
    "DIVISLAB": "Active Pharma Ingredients",
    "LUPIN": "Pharma Exporter",
    "AUROPHARMA": "Pharma Exporter",
    "TORNTPHARM": "Pharma Exporter",
    "ALKEM": "Pharma Exporter",
    "BIOCON": "Biopharma Exporter",
    "GLENMARK": "Pharma Exporter",
}

RATE_AND_BANKING_CORE = {
    "HDFCBANK": "Private Bank",
    "ICICIBANK": "Private Bank",
    "SBIN": "Public Sector Bank",
    "KOTAKBANK": "Private Bank",
    "AXISBANK": "Private Bank",
    "INDUSINDBK": "Private Bank",
    "BAJFINANCE": "Retail NBFC",
    "BAJAJFINSV": "Financial Conglomerate",
    "CHOLAFIN": "Vehicle Finance",
    "SHRIRAMFIN": "Commercial NBFC",
}


class MacroMarketEngine:
    """
    Synthesizes real-time and historical multi-asset macroeconomic indicators
    to establish the prevailing market regime and sector-specific macro forces.
    """

    @classmethod
    def analyze_macro_regime(
        cls,
        benchmark_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Computes a 360-degree Macro Market Environment Score (-100 to +100)
        and categorizes broader market health, VIX volatility risk, and commodity forces.
        """
        # 1. USD/INR exchange rate
        usdinr_rate = UniverseManager.get_usdinr_rate()
        
        # 2. Extract benchmark series
        benchmarks = benchmark_dict or {}
        def _get_df(syms):
            for s in syms:
                d = benchmarks.get(s)
                if d is not None and isinstance(d, pd.DataFrame) and not d.empty:
                    return d
            return None

        nifty_df = _get_df(["NIFTY", "^NSEI", "NIFTY50"])
        bank_df = _get_df(["BANKNIFTY", "^NSEBANK"])
        vix_df = _get_df(["INDIAVIX", "^INDIAVIX"])
        crude_df = _get_df(["CRUDEOIL", "CL=F"])
        gold_df = _get_df(["GOLD", "GC=F"])

        # Fallback fetch if not present in benchmark_dict
        needed_missing = []
        if nifty_df is None or len(nifty_df) < 10: needed_missing.append("NIFTY")
        if bank_df is None or len(bank_df) < 10: needed_missing.append("BANKNIFTY")
        if vix_df is None or len(vix_df) < 5: needed_missing.append("INDIAVIX")
        if crude_df is None or len(crude_df) < 10: needed_missing.append("CRUDEOIL")
        if gold_df is None or len(gold_df) < 10: needed_missing.append("GOLD")

        if needed_missing:
            try:
                from providers.yfinance_provider import YahooFinanceProvider
                prov = YahooFinanceProvider(cache_ttl_hours=2.0)
                fetched = prov.fetch_batch_ohlcv(needed_missing, period="3mo", interval="1d", max_workers=5)
                if "NIFTY" in fetched and fetched["NIFTY"] is not None: nifty_df = fetched["NIFTY"]
                if "BANKNIFTY" in fetched and fetched["BANKNIFTY"] is not None: bank_df = fetched["BANKNIFTY"]
                if "INDIAVIX" in fetched and fetched["INDIAVIX"] is not None: vix_df = fetched["INDIAVIX"]
                if "CRUDEOIL" in fetched and fetched["CRUDEOIL"] is not None: crude_df = fetched["CRUDEOIL"]
                if "GOLD" in fetched and fetched["GOLD"] is not None: gold_df = fetched["GOLD"]
            except Exception:
                pass

        # --- A. NIFTY 50 Analysis ---
        n_close, n_chg, n_ema20, n_ema50, n_trend, n_rsi = 22600.0, 0.0, 22500.0, 22300.0, "BULLISH", 54.0
        if nifty_df is not None and len(nifty_df) >= 5:
            n_close = float(nifty_df['close'].iloc[-1])
            prev_c = float(nifty_df['close'].iloc[-2]) if len(nifty_df) >= 2 else n_close
            n_chg = ((n_close - prev_c) / (prev_c + 1e-6)) * 100.0
            n_ema20 = float(nifty_df['close'].ewm(span=20).mean().iloc[-1])
            n_ema50 = float(nifty_df['close'].ewm(span=50).mean().iloc[-1])
            
            # Simple RSI
            delta = nifty_df['close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-6)
            rsi_series = 100 - (100 / (1 + rs))
            n_rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            if n_close >= n_ema20 and n_ema20 >= n_ema50:
                n_trend = "STRONG_BULLISH"
            elif n_close >= n_ema50:
                n_trend = "MILD_BULLISH"
            elif n_close < n_ema50 and n_chg <= -0.5:
                n_trend = "CORRECTIVE_DIP"
            else:
                n_trend = "CONSOLIDATION"

        # --- B. BANK NIFTY Analysis ---
        b_close, b_chg, b_trend = 53800.0, 0.0, "NEUTRAL"
        if bank_df is not None and len(bank_df) >= 5:
            b_close = float(bank_df['close'].iloc[-1])
            prev_b = float(bank_df['close'].iloc[-2]) if len(bank_df) >= 2 else b_close
            b_chg = ((b_close - prev_b) / (prev_b + 1e-6)) * 100.0
            b_ema20 = float(bank_df['close'].ewm(span=20).mean().iloc[-1])
            b_trend = "BULLISH" if b_close >= b_ema20 else "BEARISH"

        # --- C. INDIA VIX Volatility Gauge ---
        vix_lvl, vix_regime, vix_risk = 14.0, "NORMAL_VOLATILITY", "MODERATE"
        if vix_df is not None and len(vix_df) >= 1:
            vix_lvl = float(vix_df['close'].iloc[-1])
        
        if vix_lvl < 13.0:
            vix_regime = "LOW_VOLATILITY (COMPLACENT / TRENDING)"
            vix_risk = "LOW"
            vix_guidance = "Favorable environment for momentum breakouts and positional swings. Low chance of intraday whipsaws."
        elif vix_lvl <= 16.5:
            vix_regime = "NORMAL_VOLATILITY (HEALTHY ACTIVE TRADING)"
            vix_risk = "MODERATE"
            vix_guidance = "Healthy two-way market liquidity. Maintain standard risk:reward (1:2+) and normal position sizing."
        elif vix_lvl <= 21.0:
            vix_regime = "ELEVATED_VOLATILITY (CHOPPY / CAUTION)"
            vix_risk = "ELEVATED"
            vix_guidance = "Increased turbulence. Trim trade position sizes by 25-30%. Avoid high-beta breakouts; favor dip leaders with confirmed support."
        else:
            vix_regime = "EXTREME_VOLATILITY (PANIC / CRISIS)"
            vix_risk = "HIGH"
            vix_guidance = "Extreme fear. High failure rate for long momentum. Cut position size by 50%. Focus on shorts or high-conviction oversold rubber-band reversals."

        # --- D. CRUDE OIL Macro Analysis ---
        crude_mcx = 8870.0
        crude_usd = crude_mcx / (usdinr_rate or 96.0)
        crude_chg = 0.0
        if crude_df is not None and len(crude_df) >= 2:
            crude_mcx = float(crude_df['close'].iloc[-1])
            prev_crude = float(crude_df['close'].iloc[-2])
            crude_chg = ((crude_mcx - prev_crude) / (prev_crude + 1e-6)) * 100.0
            crude_usd = crude_mcx / (usdinr_rate or 96.0)

        if crude_usd >= 86.0 or crude_chg >= 2.0:
            crude_regime = "ELEVATED / SURGING CRUDE"
            crude_impact = "🔴 Severe cost-inflation headwind for Paints, Tyres, Aviation, and OMCs. Major windfall for ONGC and Upstream Oil."
            crude_bias = "HEADWIND_CONSUMERS"
        elif crude_usd <= 73.0:
            crude_regime = "COOLING / LOW CRUDE"
            crude_impact = "🟢 Broad margin expansion tailwind for Paints, Tyres, Auto, Chemicals, and FMCG. Muted for Upstream Oil."
            crude_bias = "TAILWIND_CONSUMERS"
        else:
            crude_regime = "STABLE / RANGEBOUND CRUDE"
            crude_impact = "🟡 Neutral impact on corporate input margins. Normal pricing dynamics apply."
            crude_bias = "NEUTRAL"

        # --- E. USD/INR Rupee Dynamics ---
        if usdinr_rate >= 92.0:
            fx_regime = "RUPEE_DEPRECIATION (ELEVATED USD)"
            fx_impact = "🌊 Strong margin realization tailwind for IT Software Exporters (TCS, INFY) and Pharma Exporters (Sun Pharma, Dr Reddy). Mild import inflation for capital goods."
            fx_bias = "TAILWIND_EXPORTERS"
        elif usdinr_rate <= 83.5:
            fx_regime = "RUPEE_APPRECIATION (STRONG RUPEE)"
            fx_impact = "🏦 Strong domestic currency boosts FII equity inflows and banking credit sentiment."
            fx_bias = "TAILWIND_DOMESTIC"
        else:
            fx_regime = "BALANCED_FX_RANGE"
            fx_impact = "⚖️ Normal currency stability. Export and domestic sectors trading on fundamentals."
            fx_bias = "NEUTRAL"

        # --- F. GOLD Safe-Haven Sentiment ---
        gold_mcx = 141400.0
        gold_chg = 0.0
        if gold_df is not None and len(gold_df) >= 2:
            gold_mcx = float(gold_df['close'].iloc[-1])
            prev_gold = float(gold_df['close'].iloc[-2])
            gold_chg = ((gold_mcx - prev_gold) / (prev_gold + 1e-6)) * 100.0

        gold_safe_haven = "Surging safe-haven demand reflects global geopolitical / inflation hedging." if gold_chg >= 1.5 else "Stable precious metals action."

        # --- G. Composite Macro Score Calculation (-100 to +100) ---
        score = 0.0
        # Nifty trend & change contribution (+- 40 pts)
        if n_trend == "STRONG_BULLISH": score += 35.0
        elif n_trend == "MILD_BULLISH": score += 20.0
        elif n_trend == "CORRECTIVE_DIP": score -= 15.0
        else: score += 5.0
        score += np.clip(n_chg * 15.0, -25.0, 25.0)

        # VIX contribution (+- 25 pts)
        if vix_risk == "LOW": score += 20.0
        elif vix_risk == "MODERATE": score += 10.0
        elif vix_risk == "ELEVATED": score -= 15.0
        else: score -= 30.0

        # Crude contribution (+- 20 pts)
        if crude_bias == "HEADWIND_CONSUMERS": score -= 15.0
        elif crude_bias == "TAILWIND_CONSUMERS": score += 15.0

        # BankNifty confirmation (+- 15 pts)
        if b_trend == "BULLISH": score += 10.0
        else: score -= 10.0

        score = float(np.clip(score, -100.0, 100.0))

        # Overall Macro Regime Badge
        if score >= 40.0:
            regime_code = "RISK_ON"
            regime_badge = "🟢 RISK-ON / MACRO TAILWIND"
            regime_desc = (
                f"Broader market is in a strong uptrend (NIFTY {n_chg:+.2f}%, above 20 EMA). "
                f"India VIX is controlled ({vix_lvl:.1f}), favoring high-conviction breakout continuation and positional swings."
            )
            regime_color = "#00E676"
        elif score >= 5.0:
            regime_code = "CAUTIOUS_RANGEBOUND"
            regime_badge = "🟡 CAUTIOUS / RANGEBOUND MACRO"
            regime_desc = (
                f"Market is consolidating with mixed cross-asset signals (NIFTY {n_chg:+.2f}%, Crude {crude_chg:+.1f}%). "
                f"Selectivity is vital: prioritize leaders with sector tailwinds and strict stop losses."
            )
            regime_color = "#FFD54F"
        elif n_chg <= -0.40 and n_rsi <= 45.0:
            regime_code = "DIP_OPPORTUNITY"
            regime_badge = "💎 MARKET DIP / COILED SPRING ACCUMULATION"
            regime_desc = (
                f"Market is undergoing an intraday/swing shakeout (NIFTY {n_chg:+.2f}%). "
                f"Broad liquidations drag all stocks, creating elite risk-reward to accumulate resilient leaders that recover first."
            )
            regime_color = "#00B0FF"
        else:
            regime_code = "RISK_OFF"
            regime_badge = "🔴 RISK-OFF / MACRO HEADWIND"
            regime_desc = (
                f"Macro pressure is elevated (Crude: ${crude_usd:.1f}/bbl, VIX: {vix_lvl:.1f}, NIFTY: {n_chg:+.2f}%). "
                f"Defensive stance recommended. Reduce position size by 30%, avoid weak high-beta, favor defensive IT/Pharma."
            )
            regime_color = "#FF5252"

        return {
            "score": round(score, 1),
            "macro_score": round(score, 1),
            "regime_code": regime_code,
            "macro_regime": regime_code,
            "regime_badge": regime_badge,
            "macro_regime_badge": regime_badge,
            "regime_color": regime_color,
            "regime_desc": regime_desc,
            "trading_playbook": regime_desc,
            "nifty": {
                "close": round(n_close, 1),
                "change_pct": round(n_chg, 2),
                "ema_20": round(n_ema20, 1),
                "ema_50": round(n_ema50, 1),
                "trend": n_trend,
                "rsi": round(n_rsi, 1)
            },
            "nifty_summary": {
                "close": round(n_close, 1),
                "change_pct": round(n_chg, 2),
                "trend": n_trend,
                "rsi": round(n_rsi, 1)
            },
            "banknifty": {
                "close": round(b_close, 1),
                "change_pct": round(b_chg, 2),
                "trend": b_trend
            },
            "vix": {
                "level": round(vix_lvl, 2),
                "risk_tier": vix_risk,
                "risk_level": vix_risk,
                "regime": vix_regime,
                "guidance": vix_guidance
            },
            "vix_summary": {
                "level": round(vix_lvl, 2),
                "regime": vix_regime,
                "risk_level": vix_risk
            },
            "crude": {
                "usd_price": round(crude_usd, 2),
                "mcx_price": round(crude_mcx, 1),
                "change_pct": round(crude_chg, 2),
                "regime": crude_regime,
                "bias": crude_bias,
                "impact_summary": crude_impact,
                "trend": crude_regime
            },
            "crude_summary": {
                "price_usd": round(crude_usd, 2),
                "price_mcx_inr": round(crude_mcx, 1),
                "change_pct": round(crude_chg, 2),
                "trend": crude_regime,
                "impact_consumers": "HEADWIND" if crude_bias == "HEADWIND_CONSUMERS" else ("TAILWIND" if crude_bias == "TAILWIND_CONSUMERS" else "NEUTRAL"),
                "impact_upstream": "TAILWIND" if crude_bias == "HEADWIND_CONSUMERS" else "NEUTRAL"
            },
            "usdinr": {
                "rate": round(usdinr_rate, 2),
                "regime": fx_regime,
                "bias": fx_bias,
                "impact_summary": fx_impact
            },
            "usdinr_summary": {
                "rate": round(usdinr_rate, 2),
                "impact_exporters": "POSITIVE" if usdinr_rate >= 83.5 else "NEUTRAL"
            },
            "gold": {
                "mcx_price": round(gold_mcx, 1),
                "change_pct": round(gold_chg, 2),
                "safe_haven_summary": gold_safe_haven
            },
            "gold_summary": {
                "mcx_price": round(gold_mcx, 1),
                "change_pct": round(gold_chg, 2)
            }
        }

    @classmethod
    def get_stock_macro_impact(
        cls,
        symbol: str,
        macro_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates how prevailing macroeconomic conditions (Crude Oil, USD/INR, VIX, NIFTY)
        act as tailwinds or headwinds for a specific stock ticker.
        """
        clean = UniverseManager.to_clean_symbol(symbol)
        if not macro_summary:
            macro_summary = cls.analyze_macro_regime()

        crude = macro_summary.get("crude", {})
        usdinr = macro_summary.get("usdinr", {})
        vix = macro_summary.get("vix", {})
        nifty = macro_summary.get("nifty", {})

        crude_bias = crude.get("bias", "NEUTRAL")
        crude_usd = crude.get("usd_price", 80.0)

        has_tailwind = False
        has_headwind = False
        conviction_delta = 0.0
        badge = "⚖️ Neutral Macro Correlation"
        thesis = "Stock performance is driven predominantly by company-specific technicals and sector flows."

        # 1. Crude-Sensitive Consumer (Paints, Tyres, Aviation, OMCs, Chemicals, Auto)
        if clean in CRUDE_SENSITIVE_CONSUMERS:
            subsector = CRUDE_SENSITIVE_CONSUMERS[clean]
            if crude_bias == "HEADWIND_CONSUMERS":
                has_headwind = True
                conviction_delta = -15.0
                badge = f"⚠️ Crude Headwind ({subsector})"
                thesis = (
                    f"Macro Headwind: Elevated crude oil (${crude_usd:.1f}/bbl) inflates input feedstock & packaging costs "
                    f"for {subsector}, threatening quarterly gross margins. Exercise discipline with tighter stop losses."
                )
            elif crude_bias == "TAILWIND_CONSUMERS":
                has_tailwind = True
                conviction_delta = +15.0
                badge = f"🌊 Crude Tailwinds ({subsector})"
                thesis = (
                    f"Macro Tailwind: Cooling crude (${crude_usd:.1f}/bbl) provides operating margin expansion and raw-material savings "
                    f"for {subsector} companies."
                )
            else:
                badge = f"⚖️ Balanced Crude ({subsector})"

        # 2. Upstream Energy Beneficiaries (ONGC, OIL)
        elif clean in CRUDE_UPSTREAM_BENEFICIARIES:
            subsector = CRUDE_UPSTREAM_BENEFICIARIES[clean]
            if crude_bias == "HEADWIND_CONSUMERS":  # High crude is a WINDFALL for upstream!
                has_tailwind = True
                conviction_delta = +20.0
                badge = f"🔥 Crude Surge Windfall ({subsector})"
                thesis = (
                    f"Macro Super-Tailwind: Surging crude (${crude_usd:.1f}/bbl) directly expands per-barrel net realization "
                    f"and cash flow for {clean}."
                )
            elif crude_bias == "TAILWIND_CONSUMERS":  # Low crude hurts upstream realization
                has_headwind = True
                conviction_delta = -15.0
                badge = f"⚠️ Weak Crude Realization ({subsector})"
                thesis = f"Macro Headwind: Subdued crude (${crude_usd:.1f}/bbl) dampens realizations for {subsector}."

        # 3. IT & Pharma Exporters (USD/INR Beneficiaries)
        elif clean in USDINR_EXPORT_BENEFICIARIES:
            subsector = USDINR_EXPORT_BENEFICIARIES[clean]
            fx_rate = usdinr.get("rate", 96.0)
            if fx_rate >= 88.0:
                has_tailwind = True
                conviction_delta = +12.0
                badge = f"💵 USD/INR FX Tailwind ({subsector})"
                thesis = (
                    f"Macro Tailwind: Elevated USD/INR (₹{fx_rate:.2f}) provides an automatic rupee realization boost "
                    f"and operating margin cushion on dollar-denominated contracts."
                )
            else:
                badge = f"⚖️ Stable FX ({subsector})"

        # 4. Rate-Sensitive Banking & NBFCs
        elif clean in RATE_AND_BANKING_CORE:
            subsector = RATE_AND_BANKING_CORE[clean]
            n_trend = nifty.get("trend", "BULLISH")
            v_risk = vix.get("risk_tier", "LOW")
            if "BULLISH" in n_trend and v_risk in ["LOW", "MODERATE"]:
                has_tailwind = True
                conviction_delta = +10.0
                badge = f"🏦 Liquidity & Credit Tailwind ({subsector})"
                thesis = f"Macro Tailwind: Broad market liquidity and healthy risk sentiment support private credit expansion for {clean}."
            elif v_risk in ["ELEVATED", "HIGH"]:
                has_headwind = True
                conviction_delta = -10.0
                badge = f"⚠️ Market Volatility Headwind ({subsector})"
                thesis = f"Macro Headwind: Elevated market volatility (VIX {vix.get('level', 15):.1f}) triggers institutional rotation and FII trimming in high-beta financials."

        # 5. General Broad Market Correlation
        else:
            n_chg = nifty.get("change_pct", 0.0)
            if n_chg <= -0.75:
                has_headwind = True
                conviction_delta = -8.0
                badge = "⚠️ Broad Market Correction"
                thesis = f"Macro Caution: Broad market selloff (NIFTY {n_chg:+.2f}%) exerts index weight drag. Only high relative strength setups should be traded."
            elif n_chg >= +0.75:
                has_tailwind = True
                conviction_delta = +8.0
                badge = "🌊 Broad Market Tailwind"
                thesis = f"Macro Tailwind: Strong benchmark momentum (NIFTY {n_chg:+.2f}%) provides positive beta lift across active setups."

        return {
            "symbol": clean,
            "has_tailwind": has_tailwind,
            "has_headwind": has_headwind,
            "badge": badge,
            "conviction_delta": conviction_delta,
            "thesis": thesis
        }
