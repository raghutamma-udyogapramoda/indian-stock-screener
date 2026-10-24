"""
Options Analytics, Greeks Engine & Multi-Leg Strategy Payoff Simulator.
Provides real-time Black-Scholes option pricing, Greeks (Delta, Theta, Gamma, Vega),
Implied Volatility (IV) solving, Max Pain pin estimation, and strategy payoff modeling
(Bull Call Spread, Bear Put Spread, Straddle, Strangle) for Indian Equities and Indices.
"""

from datetime import datetime, timezone, timedelta
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from core.fno_tracker import FNOTracker, _get_warmed_nse_session


class BlackScholesEngine:
    """
    Pure Python high-performance Black-Scholes analytical model for European options.
    Zero external C-dependency, resilient against missing platform packages.
    """

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative distribution function for standard normal distribution."""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    @staticmethod
    def norm_pdf(x: float) -> float:
        """Probability density function for standard normal distribution."""
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def price(
        cls,
        spot: float,
        strike: float,
        t_years: float,
        r: float = 0.07,
        iv: float = 0.15,
        opt_type: str = "CE"
    ) -> float:
        """Calculates theoretical option premium."""
        if spot <= 0 or strike <= 0:
            return 0.0
        if t_years <= 0 or iv <= 0:
            intrinsic = (spot - strike) if opt_type.upper() == "CE" else (strike - spot)
            return max(0.0, intrinsic)

        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
        d2 = d1 - iv * math.sqrt(t_years)

        if opt_type.upper() == "CE":
            return spot * cls.norm_cdf(d1) - strike * math.exp(-r * t_years) * cls.norm_cdf(d2)
        else:
            return strike * math.exp(-r * t_years) * cls.norm_cdf(-d2) - spot * cls.norm_cdf(-d1)

    @classmethod
    def calc_greeks(
        cls,
        spot: float,
        strike: float,
        t_years: float,
        r: float = 0.07,
        iv: float = 0.15,
        opt_type: str = "CE"
    ) -> Dict[str, float]:
        """
        Calculates option sensitivity Greeks:
          - Delta (Directional speed: 0.0 to 1.0 for CE, -1.0 to 0.0 for PE)
          - Gamma (Delta acceleration per ₹1 underlying movement)
          - Theta (Daily time decay in ₹ per day)
          - Vega (Sensitivity per 1% change in Implied Volatility)
        """
        if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
        d2 = d1 - iv * sqrt_t
        pdf_d1 = cls.norm_pdf(d1)

        gamma = pdf_d1 / (spot * iv * sqrt_t)
        vega = (spot * sqrt_t * pdf_d1) / 100.0  # per 1% IV change

        if opt_type.upper() == "CE":
            delta = cls.norm_cdf(d1)
            # 1-day theta decay
            theta = (- (spot * pdf_d1 * iv) / (2.0 * sqrt_t) - r * strike * math.exp(-r * t_years) * cls.norm_cdf(d2)) / 365.0
        else:
            delta = cls.norm_cdf(d1) - 1.0
            theta = (- (spot * pdf_d1 * iv) / (2.0 * sqrt_t) + r * strike * math.exp(-r * t_years) * cls.norm_cdf(-d2)) / 365.0

        return {
            "delta": round(delta, 3),
            "gamma": round(gamma, 5),
            "theta": round(theta, 2),
            "vega": round(vega, 2)
        }

    @classmethod
    def calc_iv(
        cls,
        market_price: float,
        spot: float,
        strike: float,
        t_years: float,
        r: float = 0.07,
        opt_type: str = "CE"
    ) -> float:
        """
        Calculates Implied Volatility (IV) in percentage (%) from market price.
        Uses hybrid Newton-Raphson with bisection fallback for rapid convergence.
        """
        if market_price <= 0 or spot <= 0 or strike <= 0 or t_years <= 0:
            return 15.0

        intrinsic = max(0.0, (spot - strike) if opt_type.upper() == "CE" else (strike - spot))
        if market_price <= intrinsic:
            return 8.0  # minimum floor for deep ITM

        # 1. Newton-Raphson iteration
        iv = 0.20
        for _ in range(12):
            price_guess = cls.price(spot, strike, t_years, r, iv, opt_type)
            diff = price_guess - market_price
            if abs(diff) < 0.05:
                return round(iv * 100.0, 1)

            sqrt_t = math.sqrt(t_years)
            d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * sqrt_t)
            vega = spot * sqrt_t * cls.norm_pdf(d1)
            if vega < 1e-4:
                break
            iv -= diff / vega
            if iv <= 0.01 or iv > 4.0:
                break

        # 2. Bisection fallback
        low, high = 0.01, 3.5
        for _ in range(20):
            mid = (low + high) / 2.0
            p = cls.price(spot, strike, t_years, r, mid, opt_type)
            if abs(p - market_price) < 0.05:
                return round(mid * 100.0, 1)
            if p > market_price:
                high = mid
            else:
                low = mid

        return round(mid * 100.0, 1)


class OptionChainAnalyzer:
    """
    Parses full option chains, extracts Greeks, Open Interest depth, and Max Pain pinning.
    """

    @classmethod
    def get_option_chain_ladder(
        cls,
        symbol: str,
        expiry: Optional[str] = None,
        strike_range_pct: float = 0.08
    ) -> Dict[str, Any]:
        """
        Retrieves real-time dual-sided Option Chain ladder for an underlying.
        Returns Calls & Puts aligned around At-The-Money (ATM).
        """
        clean_sym = FNOTracker.clean_symbol(symbol)
        quote = FNOTracker.fetch_fno_quote(clean_sym, use_cache=True)
        raw_options = quote.get("options_data", [])
        spot = quote.get("spot_cmp", 100.0)

        # 1. If options_data empty from cache, try direct fetch
        if not raw_options:
            try:
                session = _get_warmed_nse_session()
                url = f"https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi?functionName=getSymbolDerivativesData&symbol={clean_sym}"
                resp = session.get(url, timeout=7)
                if resp.status_code == 200:
                    raw_options = resp.json().get("data", [])
            except Exception:
                raw_options = []

        # Available expiry dates
        expiries = []
        seen = set()
        for x in raw_options:
            exp_date = str(x.get("expiryDate", "")).strip()
            if exp_date and exp_date not in seen:
                seen.add(exp_date)
                expiries.append(exp_date)

        chosen_expiry = expiry if (expiry and expiry in expiries) else (expiries[0] if expiries else "Near-Month")

        # Time to expiry in years
        t_years = 7.0 / 365.0  # default 7 days
        try:
            exp_dt = datetime.strptime(chosen_expiry, "%d-%b-%Y")
            now_dt = datetime.now()
            days = max(1, (exp_dt - now_dt).days)
            t_years = max(0.003, days / 365.0)
        except Exception:
            pass

        # Filter records for chosen expiry
        filtered_records = [x for x in raw_options if str(x.get("expiryDate", "")).strip() == chosen_expiry]

        # Group by strike price
        strikes_map: Dict[float, Dict[str, Any]] = {}
        for r in filtered_records:
            ot = str(r.get("optionType", "")).upper()
            if ot not in ("CE", "PE"):
                continue
            try:
                sp = float(str(r.get("strikePrice", "0")).strip())
                if sp <= 0:
                    continue
            except Exception:
                continue

            if sp not in strikes_map:
                strikes_map[sp] = {"strike": sp, "CE": None, "PE": None}

            strikes_map[sp][ot] = r

        # If empty (fallback synthetic strikes)
        if not strikes_map:
            strikes_map = cls._generate_synthetic_chain(spot, clean_sym)

        # Sort strikes
        all_sorted_strikes = sorted(list(strikes_map.keys()))
        if not all_sorted_strikes:
            return {"error": "No option strikes available"}

        # Find ATM strike
        atm_strike = min(all_sorted_strikes, key=lambda k: abs(k - spot))

        # Trim strikes within strike_range_pct (e.g. +/- 8% of spot)
        lower_bound = spot * (1.0 - strike_range_pct)
        upper_bound = spot * (1.0 + strike_range_pct)
        display_strikes = [k for k in all_sorted_strikes if lower_bound <= k <= upper_bound]
        if len(display_strikes) < 7:
            # Fallback to closest 15 strikes
            idx_atm = all_sorted_strikes.index(atm_strike)
            start_i = max(0, idx_atm - 7)
            end_i = min(len(all_sorted_strikes), idx_atm + 8)
            display_strikes = all_sorted_strikes[start_i:end_i]

        # Build Ladder Rows
        ladder_rows = []
        ce_oi_map = {}
        pe_oi_map = {}

        for sp in display_strikes:
            item = strikes_map[sp]
            ce_data = item.get("CE") or {}
            pe_data = item.get("PE") or {}

            ce_ltp = float(ce_data.get("lastPrice", 0.0))
            pe_ltp = float(pe_data.get("lastPrice", 0.0))
            ce_oi = int(ce_data.get("openInterest", 0))
            pe_oi = int(pe_data.get("openInterest", 0))
            ce_oi_chg = float(ce_data.get("pchangeinOpenInterest", 0.0))
            pe_oi_chg = float(pe_data.get("pchangeinOpenInterest", 0.0))
            ce_vol = int(ce_data.get("totalTradedVolume", 0))
            pe_vol = int(pe_data.get("totalTradedVolume", 0))

            ce_oi_map[sp] = ce_oi
            pe_oi_map[sp] = pe_oi

            # Calculate IV & Greeks
            ce_iv = BlackScholesEngine.calc_iv(ce_ltp, spot, sp, t_years, opt_type="CE") if ce_ltp > 0 else 15.0
            pe_iv = BlackScholesEngine.calc_iv(pe_ltp, spot, sp, t_years, opt_type="PE") if pe_ltp > 0 else 15.0

            ce_greeks = BlackScholesEngine.calc_greeks(spot, sp, t_years, iv=ce_iv / 100.0, opt_type="CE")
            pe_greeks = BlackScholesEngine.calc_greeks(spot, sp, t_years, iv=pe_iv / 100.0, opt_type="PE")

            ladder_rows.append({
                "strike": sp,
                "is_atm": (sp == atm_strike),
                # Call Side
                "call_oi": ce_oi,
                "call_oi_chg": ce_oi_chg,
                "call_volume": ce_vol,
                "call_iv": ce_iv,
                "call_delta": ce_greeks["delta"],
                "call_theta": ce_greeks["theta"],
                "call_ltp": ce_ltp,
                # Put Side
                "put_ltp": pe_ltp,
                "put_delta": pe_greeks["delta"],
                "put_theta": pe_greeks["theta"],
                "put_iv": pe_iv,
                "put_volume": pe_vol,
                "put_oi_chg": pe_oi_chg,
                "put_oi": pe_oi
            })

        # Max Pain Calculation
        max_pain_val = cls.calc_max_pain(display_strikes, ce_oi_map, pe_oi_map)

        tot_ce_oi = sum(ce_oi_map.values())
        tot_pe_oi = sum(pe_oi_map.values())
        pcr = round(tot_pe_oi / tot_ce_oi, 2) if tot_ce_oi > 0 else 1.0

        # ATM Row
        atm_row = next((r for r in ladder_rows if r["is_atm"]), ladder_rows[len(ladder_rows)//2] if ladder_rows else {})
        atm_iv = round((atm_row.get("call_iv", 15.0) + atm_row.get("put_iv", 15.0)) / 2.0, 1)

        return {
            "symbol": clean_sym,
            "spot": spot,
            "expiry": chosen_expiry,
            "expiries": expiries or [chosen_expiry],
            "atm_strike": atm_strike,
            "atm_iv": atm_iv,
            "max_pain": max_pain_val,
            "pcr": pcr,
            "total_call_oi": tot_ce_oi,
            "total_put_oi": tot_pe_oi,
            "ladder": ladder_rows
        }

    @staticmethod
    def calc_max_pain(strikes: List[float], ce_oi: Dict[float, int], pe_oi: Dict[float, int]) -> float:
        """
        Computes Max Pain strike: The strike price at which total option writers' payout
        is minimized (or option buyers' loss is maximized).
        """
        if not strikes:
            return 0.0

        pains: Dict[float, float] = {}
        for s_target in strikes:
            ce_loss = sum(max(0.0, s_target - k) * oi for k, oi in ce_oi.items())
            pe_loss = sum(max(0.0, k - s_target) * oi for k, oi in pe_oi.items())
            pains[s_target] = ce_loss + pe_loss

        return min(pains, key=pains.get)

    @staticmethod
    def _generate_synthetic_chain(spot: float, symbol: str) -> Dict[float, Dict[str, Any]]:
        """Generates realistic synthetic strikes when live option chain is in offline test mode."""
        step = 50.0 if spot > 1500 else (20.0 if spot > 500 else 10.0)
        base = round(spot / step) * step
        strikes_map = {}
        for i in range(-10, 11):
            sp = base + (i * step)
            # Rough approximation of prices
            ce_price = max(0.5, (spot - sp) + 30.0 * math.exp(-abs(i) * 0.25)) if sp >= spot else max(spot - sp + 5.0, 5.0)
            pe_price = max(0.5, (sp - spot) + 30.0 * math.exp(-abs(i) * 0.25)) if sp <= spot else max(sp - spot + 5.0, 5.0)
            ce_oi = int(max(1000, 50000 * math.exp(-abs(i - 2) * 0.3)))
            pe_oi = int(max(1000, 50000 * math.exp(-abs(i + 2) * 0.3)))

            strikes_map[sp] = {
                "strike": sp,
                "CE": {"lastPrice": round(ce_price, 2), "openInterest": ce_oi, "pchangeinOpenInterest": 5.0, "totalTradedVolume": ce_oi // 2},
                "PE": {"lastPrice": round(pe_price, 2), "openInterest": pe_oi, "pchangeinOpenInterest": -3.0, "totalTradedVolume": pe_oi // 2}
            }
        return strikes_map


class StrategyPayoffEngine:
    """
    Simulates multi-leg derivative options strategies and computes payoff curves.
    """

    @classmethod
    def generate_strategy_payoff(
        cls,
        strategy_type: str,
        spot: float,
        lot_size: int = 250,
        custom_legs: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """
        Builds payoff profile for standard strategies:
          - BULL_CALL_SPREAD (Buy ATM CE, Sell OTM CE)
          - BEAR_PUT_SPREAD (Buy ATM PE, Sell OTM PE)
          - LONG_STRADDLE (Buy ATM CE + ATM PE)
          - LONG_STRANGLE (Buy OTM CE + OTM PE)
          - LONG_CALL (Single Naked Call)
          - LONG_PUT (Single Naked Put)
        """
        # Dynamic step and pricing based on Black-Scholes model
        step = 100.0 if spot > 15000 else (50.0 if spot > 3000 else (20.0 if spot > 1000 else (10.0 if spot > 300 else 5.0)))
        atm_strike = round(spot / step) * step
        t_ref = 7.0 / 365.0  # standard 7-day expiry horizon
        iv_ref = 0.14        # baseline 14% implied volatility

        legs = []
        strat_title = ""
        strat_bias = ""
        strat_desc = ""

        if strategy_type == "BULL_CALL_SPREAD":
            k1 = atm_strike
            k2 = atm_strike + step
            p1 = round(BlackScholesEngine.price(spot, k1, t_ref, iv=iv_ref, opt_type="CE"), 2)
            p2 = round(BlackScholesEngine.price(spot, k2, t_ref, iv=iv_ref, opt_type="CE"), 2)
            legs = [
                {"action": "BUY", "type": "CE", "strike": k1, "price": p1, "qty": lot_size},
                {"action": "SELL", "type": "CE", "strike": k2, "price": p2, "qty": lot_size}
            ]
            strat_title = f"Bull Call Spread ({k1:.0f} CE / {k2:.0f} CE)"
            strat_bias = "MODERATELY BULLISH"
            strat_desc = "Defined Risk & Defined Profit: Captures upside thrust while selling OTM call to finance theta decay."

        elif strategy_type == "BEAR_PUT_SPREAD":
            k1 = atm_strike - step
            k2 = atm_strike
            p1 = round(BlackScholesEngine.price(spot, k1, t_ref, iv=iv_ref, opt_type="PE"), 2)
            p2 = round(BlackScholesEngine.price(spot, k2, t_ref, iv=iv_ref, opt_type="PE"), 2)
            legs = [
                {"action": "BUY", "type": "PE", "strike": k2, "price": p2, "qty": lot_size},
                {"action": "SELL", "type": "PE", "strike": k1, "price": p1, "qty": lot_size}
            ]
            strat_title = f"Bear Put Spread ({k2:.0f} PE / {k1:.0f} PE)"
            strat_bias = "MODERATELY BEARISH"
            strat_desc = "Defined Risk Downside Hedge: Exploits institutional breakdown while capping maximum cost."

        elif strategy_type == "LONG_STRADDLE":
            p_ce = round(BlackScholesEngine.price(spot, atm_strike, t_ref, iv=iv_ref, opt_type="CE"), 2)
            p_pe = round(BlackScholesEngine.price(spot, atm_strike, t_ref, iv=iv_ref, opt_type="PE"), 2)
            legs = [
                {"action": "BUY", "type": "CE", "strike": atm_strike, "price": p_ce, "qty": lot_size},
                {"action": "BUY", "type": "PE", "strike": atm_strike, "price": p_pe, "qty": lot_size}
            ]
            strat_title = f"Long Straddle ({atm_strike:.0f} CE + PE)"
            strat_bias = "HIGH VOLATILITY BREAKOUT (DELTA-NEUTRAL)"
            strat_desc = "Explosive Volatility Play: Profits from a massive directional move in either direction (Earnings/Budget/Breakout)."

        elif strategy_type == "LONG_STRANGLE":
            k_ce = atm_strike + step
            k_pe = atm_strike - step
            p_ce = round(BlackScholesEngine.price(spot, k_ce, t_ref, iv=iv_ref, opt_type="CE"), 2)
            p_pe = round(BlackScholesEngine.price(spot, k_pe, t_ref, iv=iv_ref, opt_type="PE"), 2)
            legs = [
                {"action": "BUY", "type": "CE", "strike": k_ce, "price": p_ce, "qty": lot_size},
                {"action": "BUY", "type": "PE", "strike": k_pe, "price": p_pe, "qty": lot_size}
            ]
            strat_title = f"Long Strangle ({k_ce:.0f} CE / {k_pe:.0f} PE)"
            strat_bias = "CHEAP VOLATILITY EXPANSION"
            strat_desc = "Lower-Cost Out-of-the-Money Volatility: Requires larger percentage move to reach profitability."

        elif strategy_type == "LONG_PUT":
            p = round(BlackScholesEngine.price(spot, atm_strike, t_ref, iv=iv_ref, opt_type="PE"), 2)
            legs = [{"action": "BUY", "type": "PE", "strike": atm_strike, "price": p, "qty": lot_size}]
            strat_title = f"Naked Long Put ({atm_strike:.0f} PE)"
            strat_bias = "AGGRESSIVE BEARISH"
            strat_desc = "Directional Short Bet: Heavy theta decay risk if stock consolidates."

        else:  # LONG_CALL default
            p = round(spot * 0.022, 2)
            legs = [{"action": "BUY", "type": "CE", "strike": atm_strike, "price": p, "qty": lot_size}]
            strat_title = f"Naked Long Call ({atm_strike:.0f} CE)"
            strat_bias = "AGGRESSIVE BULLISH"
            strat_desc = "Directional Long Bet: High delta thrust but vulnerable to time erosion."

        # Compute Payoff Curves over price range (+/- 12% of spot)
        price_range = np.linspace(spot * 0.88, spot * 1.12, 100)
        pnl_points = []
        for p_at_exp in price_range:
            tot_pnl = 0.0
            for leg in legs:
                act = leg["action"]
                ot = leg["type"]
                k = leg["strike"]
                premium = leg["price"]
                q = leg["qty"]

                if ot == "CE":
                    val = max(0.0, p_at_exp - k)
                else:
                    val = max(0.0, k - p_at_exp)

                if act == "BUY":
                    leg_pnl = (val - premium) * q
                else:
                    leg_pnl = (premium - val) * q

                tot_pnl += leg_pnl

            pnl_points.append(round(tot_pnl, 2))

        pnl_arr = np.array(pnl_points)
        max_profit = float(np.max(pnl_arr))
        max_loss = float(np.min(pnl_arr))
        
        # Net Debit/Credit
        net_cash = sum(
            (-leg["price"] if leg["action"] == "BUY" else leg["price"]) * leg["qty"]
            for leg in legs
        )

        # Breakeven calculation
        zero_crossings = []
        for i in range(len(pnl_points) - 1):
            if (pnl_points[i] <= 0 and pnl_points[i+1] > 0) or (pnl_points[i] >= 0 and pnl_points[i+1] < 0):
                be_price = price_range[i]
                zero_crossings.append(round(float(be_price), 2))

        rr_ratio = abs(max_profit / max_loss) if max_loss < 0 else 999.0
        rr_str = f"1:{rr_ratio:.2f}" if max_loss < 0 else "Unlimited / Risk-Free"

        return {
            "title": strat_title,
            "bias": strat_bias,
            "description": strat_desc,
            "legs": legs,
            "spot": spot,
            "max_profit": max_profit,
            "max_profit_str": f"₹{max_profit:,.2f}" if max_profit < 1000000 else "Unlimited 🚀",
            "max_loss": max_loss,
            "max_loss_str": f"₹{abs(max_loss):,.2f}",
            "net_cash": net_cash,
            "cash_type": "NET DEBIT" if net_cash < 0 else "NET CREDIT",
            "risk_reward": rr_str,
            "breakevens": zero_crossings or [round(spot, 2)],
            "prices": [round(float(x), 2) for x in price_range],
            "pnls": pnl_points
        }
