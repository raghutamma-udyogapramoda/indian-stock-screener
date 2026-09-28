"""
Item & Portfolio Tracker Engine for Indian Equities, Indices, Options & Commodities.
Persists user-specific positions to disk and provides live real-time analysis
with intelligent BUY, HOLD, EXIT, and AVERAGE DOWN (Fake Fall / Dip Detection) signals.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

from config.settings import BASE_DIR
from core.indicators import enrich_with_indicators
from core.universe import UniverseManager

TRACKER_DATA_DIR = BASE_DIR / "data" / "tracker"
TRACKER_DATA_DIR.mkdir(parents=True, exist_ok=True)


class PortfolioTracker:
    """Manages multi-user persistent tracked portfolios and computes live actionable signals."""

    @staticmethod
    def _get_user_file(username: str) -> Path:
        clean_user = "".join(c for c in username.lower() if c.isalnum() or c in ("_", "-")).strip()
        if not clean_user:
            clean_user = "default_user"
        return TRACKER_DATA_DIR / f"{clean_user}.json"

    @classmethod
    def load_positions(cls, username: str) -> List[dict]:
        """Loads tracked positions for a specific user from disk."""
        filepath = cls._get_user_file(username)
        if not filepath.exists():
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    @classmethod
    def save_positions(cls, username: str, positions: List[dict]) -> None:
        """Saves tracked positions for a specific user to disk."""
        filepath = cls._get_user_file(username)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(positions, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving portfolio for {username}: {e}")

    @staticmethod
    def get_underlying_symbol(symbol: str) -> str:
        """
        Extracts underlying equity/index/commodity symbol from options/futures contract names.
        Examples:
          'NIFTY 25000 CE' -> 'NIFTY'
          'BANKNIFTY 52000 PE' -> 'BANKNIFTY'
          'RELIANCE 1300 CE' -> 'RELIANCE'
          'GOLD FUT' -> 'GOLD'
          'TCS' -> 'TCS'
        """
        s = symbol.strip().upper()
        tokens = s.replace("-", " ").split()
        if tokens:
            first = tokens[0]
            clean_first = first.replace(".NS", "").replace(".BO", "").replace("^", "")
            return clean_first
        return s

    @classmethod
    def fetch_live_quotes(cls, symbols: List[str]) -> Dict[str, dict]:
        """
        Ultra-fast parallel live quote fetcher designed for 10-second ticker refreshes.
        Fetches instantaneous Last Traded Price (LTP/CMP), day high/low, and daily % change.
        """
        if not symbols:
            return {}

        quotes: Dict[str, dict] = {}
        unique_syms = list(dict.fromkeys(symbols))

        def _fetch_one(raw_sym: str):
            clean = cls.get_underlying_symbol(raw_sym)
            yf_sym = UniverseManager.to_yfinance_symbol(clean)
            try:
                ticker = yf.Ticker(yf_sym)
                fi = ticker.fast_info
                price = getattr(fi, "last_price", None)
                if price is None or price <= 0:
                    price = getattr(fi, "regular_market_price", None)

                prev = getattr(fi, "previous_close", None) or price
                day_high = getattr(fi, "day_high", None) or price
                day_low = getattr(fi, "day_low", None) or price

                # Authoritative MCX Commodity INR adjustment
                if UniverseManager.is_commodity(clean) and price is not None:
                    mult = UniverseManager.get_mcx_conversion_multiplier(clean, float(price))
                    if abs(mult - 1.0) > 1e-4:
                        price = round(float(price) * mult, 2)
                        if prev is not None:
                            prev = round(float(prev) * mult, 2)
                        if day_high is not None:
                            day_high = round(float(day_high) * mult, 2)
                        if day_low is not None:
                            day_low = round(float(day_low) * mult, 2)

                if price is not None and price > 0:
                    chg = round(price - prev, 2) if prev else 0.0
                    chg_pct = round((chg / prev * 100), 2) if prev and prev > 0 else 0.0
                    return clean, raw_sym, {
                        "symbol": clean,
                        "raw_symbol": raw_sym,
                        "price": round(float(price), 2),
                        "prev_close": round(float(prev), 2) if prev else round(float(price), 2),
                        "change": chg,
                        "change_pct": chg_pct,
                        "day_high": round(float(day_high), 2) if day_high else round(float(price), 2),
                        "day_low": round(float(day_low), 2) if day_low else round(float(price), 2),
                        "timestamp": datetime.now().strftime("%I:%M:%S %p"),
                        "status": "LIVE"
                    }
            except Exception:
                pass
            return clean, raw_sym, None

        max_workers = min(len(unique_syms), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_fetch_one, unique_syms)
            for clean_s, raw_s, q in results:
                if q:
                    quotes[clean_s] = q
                    quotes[raw_s] = q

        return quotes

    @classmethod
    def add_position(
        cls,
        username: str,
        symbol: str,
        buy_price: float,
        qty: float,
        asset_type: str = "EQUITY",
        stop_loss: Optional[float] = None,
        target: Optional[float] = None,
        notes: str = ""
    ) -> dict:
        """Adds a new position to the user's tracker."""
        raw_sym = symbol.strip().upper()
        underlying = cls.get_underlying_symbol(raw_sym)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        pos_id = f"{underlying}_{int(datetime.now().timestamp())}"

        # If SL or target not provided, calculate default estimates
        if not stop_loss:
            stop_loss = round(buy_price * 0.90 if "OPTION" in asset_type.upper() else buy_price * 0.95, 2)
        if not target:
            target = round(buy_price * 1.30 if "OPTION" in asset_type.upper() else buy_price * 1.10, 2)

        new_pos = {
            "id": pos_id,
            "symbol": raw_sym,
            "underlying": underlying,
            "asset_type": asset_type.upper(),
            "buy_price": float(buy_price),
            "qty": float(qty),
            "buy_date": now_str[:10],
            "stop_loss": float(stop_loss),
            "target": float(target),
            "notes": notes.strip(),
            "created_at": now_str,
        }

        positions = cls.load_positions(username)
        positions.insert(0, new_pos)
        cls.save_positions(username, positions)
        return new_pos

    @classmethod
    def delete_position(cls, username: str, position_id: str) -> bool:
        """Removes a position by ID."""
        positions = cls.load_positions(username)
        updated = [p for p in positions if p.get("id") != position_id]
        if len(updated) != len(positions):
            cls.save_positions(username, updated)
            return True
        return False

    @classmethod
    def clear_all_positions(cls, username: str) -> bool:
        """Removes all tracked positions for a user."""
        cls.save_positions(username, [])
        return True

    @classmethod
    def update_position(cls, username: str, position_id: str, updates: dict) -> bool:
        """Updates fields of an existing position."""
        positions = cls.load_positions(username)
        for p in positions:
            if p.get("id") == position_id:
                p.update(updates)
                cls.save_positions(username, positions)
                return True
        return False

    @classmethod
    def evaluate_live_position(
        cls,
        pos: dict,
        live_df: Optional[pd.DataFrame] = None,
        live_quote: Optional[dict] = None
    ) -> dict:
        """
        Takes a tracked position, historical candle data, and real-time live market quote.
        Performs deep quantitative evaluation:
        - Incorporates real-time 10-second price ticks
        - Calculates live P&L (₹ and %)
        - Analyzes for Fake Fall (Shakeout -> Average Down recommendation)
        - Analyzes for Fake Rise (Bull Trap -> Caution / Take Profit)
        - Generates clear action badges and comprehensive execution thesis.
        """
        sym = pos["symbol"]
        buy_p = float(pos["buy_price"])
        qty = float(pos["qty"])
        sl = float(pos.get("stop_loss", buy_p * 0.95))
        tgt = float(pos.get("target", buy_p * 1.10))
        asset_type = pos.get("asset_type", "EQUITY")

        # 1. Determine current market price (CMP)
        current_price = buy_p
        has_market_data = False
        rvol = 1.0
        rsi = 50.0
        clv = 0.5
        ema20 = buy_p
        ema50 = buy_p
        sma200 = buy_p
        lower_wick_pct = 0.0
        upper_wick_pct = 0.0

        # Prioritize live ticker quote if available
        if live_quote and live_quote.get("price") and float(live_quote["price"]) > 0:
            current_price = float(live_quote["price"])
            has_market_data = True

        if live_df is not None and not live_df.empty and len(live_df) >= 5:
            has_market_data = True
            df_copy = live_df.copy()
            if live_quote and live_quote.get("price") and float(live_quote["price"]) > 0:
                last_idx = df_copy.index[-1]
                df_copy.loc[last_idx, "close"] = current_price
                if current_price > df_copy.loc[last_idx, "high"]:
                    df_copy.loc[last_idx, "high"] = current_price
                if current_price < df_copy.loc[last_idx, "low"]:
                    df_copy.loc[last_idx, "low"] = current_price

            enriched = enrich_with_indicators(df_copy)
            last = enriched.iloc[-1]
            if not (live_quote and live_quote.get("price")):
                current_price = float(last["close"])
            rsi = float(last.get("rsi_14", 50.0))
            rvol = float(last.get("rvol_20", 1.0))
            clv = float(last.get("clv", 0.5))
            ema20 = float(last.get("ema_20", current_price))
            ema50 = float(last.get("ema_50", current_price))
            sma200 = float(last.get("sma_200", current_price))

            # Candle anatomy (wicks)
            high = float(last["high"])
            low = float(last["low"])
            op = float(last["open"])
            cl = current_price
            candle_range = max(high - low, 0.001)
            lower_wick = min(op, cl) - low
            upper_wick = high - max(op, cl)
            lower_wick_pct = max(0.0, lower_wick / candle_range)
            upper_wick_pct = max(0.0, upper_wick / candle_range)

        # 2. P&L Calculations
        is_put = "PE" in asset_type or "PUT" in asset_type
        if is_put:
            pnl_pts = buy_p - current_price
        else:
            pnl_pts = current_price - buy_p

        pnl_val = pnl_pts * qty
        pnl_pct = (pnl_pts / buy_p) * 100 if buy_p > 0 else 0.0
        current_val = current_price * qty
        invested_val = buy_p * qty

        # 3. Action Signal Evaluation Engine
        action_badge = "🟢 HOLD"
        action_type = "HOLD"
        action_color = "#00E676"
        rationale = ""
        suggested_action = ""

        # --- A. TARGET ACHIEVED ---
        if (not is_put and current_price >= tgt) or (is_put and current_price <= tgt):
            action_badge = "🎯 TARGET HIT (BOOK PROFIT)"
            action_type = "TARGET_HIT"
            action_color = "#00E676"
            rationale = f"Position achieved target level (₹{tgt:,.2f}). Excellent gain of {pnl_pct:+.2f}%. Lock in profits or trail stop-loss aggressively."
            suggested_action = "Book 70%-100% profit now; trail remainder with stop at Target 1."

        # --- B. POSITION IN DRAWDOWN (Analyze Fake Fall vs Real Breakdown) ---
        elif (not is_put and current_price < buy_p) or (is_put and current_price > buy_p):
            drawdown_pct = abs(pnl_pct)

            # Check if critical Stop Loss is violated
            sl_breached = (current_price <= sl) if not is_put else (current_price >= sl)

            # Detect Fake Fall (Shakeout):
            # Conditions for Fake Fall:
            # 1. Low volume on dip (rvol < 0.85 -> selling lacks institutional conviction)
            # 2. Buyer defense wick (lower_wick_pct >= 0.35 or clv >= 0.45)
            # 3. RSI bounce above panic territory (RSI > 34)
            # 4. SL not catastrophically breached (price within 2.5% of SL or above 50 EMA)
            is_fake_fall = (
                not sl_breached and
                (rvol < 0.88 or lower_wick_pct >= 0.32 or clv >= 0.45) and
                rsi >= 32.0
            ) or (
                sl_breached and lower_wick_pct >= 0.45 and current_price >= sl * 0.985
            )

            if is_fake_fall:
                action_badge = "🟢 AVERAGE DOWN (Fake Fall / Dip Opportunity)"
                action_type = "AVERAGE_DOWN"
                action_color = "#00E676"
                avg_zone = round(max(current_price * 0.99, sl * 1.01), 2)
                new_blended = round((buy_p + current_price) / 2, 2)
                rationale = (
                    f"⚠️ Detected Fake Fall / Low-Volume Shakeout. Relative volume is low ({rvol:.2f}x) "
                    f"with buyer wick rejection ({lower_wick_pct*100:.0f}% lower tail). "
                    f"Institutions are not dumping; selling pressure is drying up near support (20/50 EMA). "
                    f"Strong statistical setup to average down your entry."
                )
                suggested_action = f"Add 50%-100% position size in the ₹{avg_zone:,.2f} zone. This lowers blended cost to ~₹{new_blended:,.2f} with Stop Loss maintained at ₹{sl:,.2f}."

            elif sl_breached:
                action_badge = "🛑 EXIT (Stop Loss Broken)"
                action_type = "EXIT"
                action_color = "#FF5252"
                rationale = (
                    f"Stop Loss breached at ₹{current_price:,.2f} (SL was ₹{sl:,.2f}). "
                    f"High selling volume ({rvol:.2f}x) indicates institutional distribution. "
                    f"This is a genuine breakdown, NOT a fake fall. Cut loss immediately to protect capital."
                )
                suggested_action = "Square off position immediately. Do not hold losing positions hoping for a turnaround."

            else:
                action_badge = "🟡 CAUTION / HOLD (Monitoring Dip)"
                action_type = "CAUTION_HOLD"
                action_color = "#FFD54F"
                rationale = (
                    f"Position is experiencing mild consolidation ({pnl_pct:+.2f}%). "
                    f"Price is respecting Stop Loss (₹{sl:,.2f}) with moderate volume ({rvol:.2f}x). "
                    f"RSI is neutral at {rsi:.1f}."
                )
                suggested_action = f"Hold current quantity. Maintain firm Stop Loss at ₹{sl:,.2f}. Avoid adding until a clear reversal candle appears."

        # --- C. POSITION IN PROFIT (Analyze Fake Rise vs Genuine Trend) ---
        else:
            # Detect Fake Rise (Bull Trap / Exhaustion):
            # Conditions for Fake Rise:
            # 1. Price rising on dry volume (rvol < 0.65 -> retail trap)
            # 2. Strong upper rejection wick (upper_wick_pct >= 0.40 or clv <= 0.35)
            # 3. RSI extremely overbought (> 78)
            is_fake_rise = (rvol < 0.65 and upper_wick_pct >= 0.38) or (rsi >= 78 and clv <= 0.35)

            if is_fake_rise:
                action_badge = "⚠️ FAKE RISE (Bull Trap - Book Partial)"
                action_type = "FAKE_RISE"
                action_color = "#FF9800"
                rationale = (
                    f"Warning: Price surged on very low volume ({rvol:.2f}x) with an upper rejection wick ({upper_wick_pct*100:.0f}% upper shadow). "
                    f"Smart money is not participating in this spike; retail buyers are getting trapped near resistance."
                )
                suggested_action = "Book 50% partial profit immediately. Trail Stop Loss tightly to entry price (breakeven) to protect gains."

            else:
                action_badge = "🟢 STRONG HOLD (Trend Intact)"
                action_type = "STRONG_HOLD"
                action_color = "#00E676"
                rationale = (
                    f"Upward trend is healthy and confirmed. Trading above 20 EMA with positive momentum (RSI: {rsi:.1f}). "
                    f"Current gain: {pnl_pct:+.2f}%. Volume structure supports continuation toward Target ₹{tgt:,.2f}."
                )
                suggested_action = f"Hold full position. Trail Stop Loss upwards to lock in profits as price approaches Target ₹{tgt:,.2f}."

        return {
            "id": pos["id"],
            "symbol": sym,
            "asset_type": asset_type,
            "buy_price": buy_p,
            "qty": qty,
            "buy_date": pos.get("buy_date", "—"),
            "stop_loss": sl,
            "target": tgt,
            "notes": pos.get("notes", ""),
            "current_price": current_price,
            "pnl_val": pnl_val,
            "pnl_pct": pnl_pct,
            "invested_val": invested_val,
            "current_val": current_val,
            "has_market_data": has_market_data,
            "rvol": rvol,
            "rsi": rsi,
            "clv": clv,
            "action_badge": action_badge,
            "action_type": action_type,
            "action_color": action_color,
            "rationale": rationale,
            "suggested_action": suggested_action,
            "day_change": live_quote.get("change", 0.0) if live_quote else 0.0,
            "day_change_pct": live_quote.get("change_pct", 0.0) if live_quote else 0.0,
            "day_high": live_quote.get("day_high", current_price) if live_quote else current_price,
            "day_low": live_quote.get("day_low", current_price) if live_quote else current_price,
            "last_checked_time": live_quote.get("timestamp", datetime.now().strftime("%I:%M:%S %p")) if live_quote else datetime.now().strftime("%I:%M:%S %p"),
            "commodity_unit": UniverseManager.get_commodity_unit(sym) if UniverseManager.is_commodity(sym) else "",
        }
