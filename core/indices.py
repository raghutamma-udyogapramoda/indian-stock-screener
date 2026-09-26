"""
Indian Market & Sectoral Indices Derivatives Hub.
Provides comprehensive algorithmic analysis for major Indian indices (NIFTY 50, BANK NIFTY, SENSEX, FIN NIFTY, MIDCAP NIFTY, NIFTY IT).
Generates Buy / Sell / Hold signals for Index Futures, explores Options setups (CE/PE/Spreads),
and ranks indices by trade profitability.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.indicators import enrich_with_indicators


# Detailed specifications for major Indian market and sectoral indices
INDEX_SPECS: Dict[str, Dict[str, Any]] = {
    "NIFTY": {
        "symbol": "NIFTY",
        "display_name": "NIFTY 50",
        "exchange": "NSE",
        "yahoo_ticker": "^NSEI",
        "lot_size": 65,
        "strike_step": 50,
        "tick_size": 0.05,
        "description": "Benchmark index of the top 50 large-cap Indian companies across 13 economic sectors.",
        "typical_spread": "Narrow (0.05 - 0.15 pts)"
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY",
        "display_name": "BANK NIFTY",
        "exchange": "NSE",
        "yahoo_ticker": "^NSEBANK",
        "lot_size": 30,
        "strike_step": 100,
        "tick_size": 0.05,
        "description": "High-beta sector index tracking 12 major public and private commercial banks.",
        "typical_spread": "Moderate (0.20 - 0.50 pts)"
    },
    "SENSEX": {
        "symbol": "SENSEX",
        "display_name": "BSE SENSEX",
        "exchange": "BSE",
        "yahoo_ticker": "^BSESN",
        "lot_size": 20,
        "strike_step": 100,
        "tick_size": 0.05,
        "description": "Premier benchmark of 30 financially sound bluechips listed on the Bombay Stock Exchange.",
        "typical_spread": "Moderate (0.50 - 1.0 pts)"
    },
    "FINNIFTY": {
        "symbol": "FINNIFTY",
        "display_name": "FIN NIFTY",
        "exchange": "NSE",
        "yahoo_ticker": "NIFTY_FIN_SERVICE.NS",
        "lot_size": 60,
        "strike_step": 50,
        "tick_size": 0.05,
        "description": "Financial services benchmark covering banking, insurance, NBFCs, and asset management.",
        "typical_spread": "Narrow (0.10 - 0.25 pts)"
    },
    "MIDCPNIFTY": {
        "symbol": "MIDCPNIFTY",
        "display_name": "MIDCAP NIFTY",
        "exchange": "NSE",
        "yahoo_ticker": "^NSEMDCP50",
        "lot_size": 120,
        "strike_step": 25,
        "tick_size": 0.05,
        "description": "Captures the performance and high-growth momentum of top 50 liquid mid-cap leaders.",
        "typical_spread": "Moderate (0.15 - 0.35 pts)"
    },
    "NIFTYIT": {
        "symbol": "NIFTYIT",
        "display_name": "NIFTY IT",
        "exchange": "NSE",
        "yahoo_ticker": "^CNXIT",
        "lot_size": 0,
        "strike_step": 100,
        "tick_size": 0.05,
        "description": "Information Technology sector tracking India's software exporters (Cash benchmark; trade via IT stock F&O).",
        "typical_spread": "Moderate (0.50 - 1.5 pts)"
    },
    "INDIAVIX": {
        "symbol": "INDIAVIX",
        "display_name": "INDIA VIX",
        "exchange": "NSE",
        "yahoo_ticker": "^INDIAVIX",
        "lot_size": 0,
        "strike_step": 0,
        "tick_size": 0.01,
        "description": "National Volatility Gauge measuring market's expectation of 30-day forward volatility.",
        "typical_spread": "N/A"
    }
}


class IndexDerivativesAnalyzer:
    """
    Analyzes Indian market indices using technical indicators to generate:
      1. Futures Trading Signals (BUY / SELL / HOLD) with exact entry, stop-loss, and targets.
      2. Comprehensive Options Strategies (Call, Put, Bull/Bear Spreads, Straddles).
      3. Index Profitability Scoring & Ranking to identify the most lucrative index setup.
    """

    @staticmethod
    def analyze_index(symbol: str, df: pd.DataFrame, india_vix: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Runs comprehensive technical and derivatives evaluation on a single index DataFrame.
        """
        if df is None or len(df) < 20:
            return None

        # Clean symbol
        clean_sym = symbol.upper().replace("^", "").replace(".NS", "").replace(".BO", "")
        if clean_sym in ["NSEI", "NIFTY50", "NIFTY_50"]:
            clean_sym = "NIFTY"
        elif clean_sym in ["NSEBANK", "BANK_NIFTY"]:
            clean_sym = "BANKNIFTY"
        elif clean_sym in ["BSESN"]:
            clean_sym = "SENSEX"
        elif clean_sym in ["NIFTY_FIN_SERVICE", "FIN_NIFTY"]:
            clean_sym = "FINNIFTY"
        elif clean_sym in ["NSEMDCP50", "MIDCAP_NIFTY"]:
            clean_sym = "MIDCPNIFTY"
        elif clean_sym in ["CNXIT", "NIFTY_IT"]:
            clean_sym = "NIFTYIT"

        specs = INDEX_SPECS.get(clean_sym, {
            "symbol": clean_sym,
            "display_name": clean_sym,
            "exchange": "NSE",
            "lot_size": 25,
            "strike_step": 50,
            "description": "Indian Market Index"
        })

        # Enrich with indicators
        df = enrich_with_indicators(df.copy())
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        close = float(last['close'])
        prev_close = float(prev['close'])
        change_pts = close - prev_close
        change_pct = (change_pts / (prev_close + 1e-6)) * 100.0

        ema9 = float(last.get('ema_9', close))
        ema20 = float(last.get('ema_20', close))
        ema50 = float(last.get('ema_50', close))
        sma200 = float(last.get('sma_200', close)) if 'sma_200' in df.columns and not np.isnan(last['sma_200']) else ema50

        rsi = float(last.get('rsi_14', 50.0))
        atr = float(last.get('atr_14', close * 0.012))
        bb_upper = float(last.get('bb_upper', close * 1.02))
        bb_lower = float(last.get('bb_lower', close * 0.98))
        bb_middle = float(last.get('bb_middle', close))
        bbw = float(last.get('bb_bandwidth', 0.03))

        macd_line = float(last.get('macd_line', 0))
        macd_signal = float(last.get('macd_signal', 0))
        macd_hist = float(last.get('macd_hist', 0))
        macd_level2 = (macd_line > macd_signal) and (macd_line > 0)

        aroon_up = float(last.get('aroon_up', 50.0))
        aroon_down = float(last.get('aroon_down', 50.0))
        aroon_osc = float(last.get('aroon_osc', 0.0))
        clv = float(last.get('clv', 0.5))

        # Check for Bollinger Squeeze
        bbw_series = df['bb_bandwidth'].iloc[-30:] if len(df) >= 30 else df['bb_bandwidth']
        is_squeezing = bbw <= (float(bbw_series.min()) * 1.25)

        # ------------------------------------------------------------------
        # 1. FUTURES SIGNAL DETERMINATION (BUY / SELL / HOLD)
        # ------------------------------------------------------------------
        bullish_score = 0
        bearish_score = 0
        technical_reasons = []

        # Moving average trend filter
        if close > ema20 and ema20 >= ema50 * 0.995:
            bullish_score += 25
            technical_reasons.append("Trading above 20 EMA and 50 EMA in Stage-2 uptrend")
        elif close < ema20 and ema20 <= ema50 * 1.005:
            bearish_score += 25
            technical_reasons.append("Trading below 20 EMA and 50 EMA in Stage-4 corrective trend")

        # SMA 200 Macro Trend
        if close >= sma200:
            bullish_score += 15
        else:
            bearish_score += 15
            technical_reasons.append("Trading below 200 SMA long-term benchmark")

        # RSI Momentum
        if 54.0 <= rsi <= 68.0:
            bullish_score += 20
            technical_reasons.append(f"Bullish RSI momentum expansion ({rsi:.1f})")
        elif rsi > 72.0:
            bullish_score += 5
            technical_reasons.append(f"RSI overbought ({rsi:.1f}), caution on fresh longs")
        elif 32.0 <= rsi <= 46.0:
            bearish_score += 20
            technical_reasons.append(f"Bearish RSI momentum breakdown ({rsi:.1f})")
        elif rsi < 28.0:
            bearish_score += 5
            technical_reasons.append(f"RSI deeply oversold ({rsi:.1f}), short squeeze possible")

        # MACD & Level-II formation
        if macd_level2:
            bullish_score += 20
            technical_reasons.append("MACD Level-II bullish continuation (MACD > Signal above zero line)")
        elif macd_hist > 0:
            bullish_score += 10
            technical_reasons.append("MACD bullish histogram expansion")
        elif (macd_line < macd_signal) and (macd_line < 0):
            bearish_score += 20
            technical_reasons.append("MACD bearish continuation below zero line")
        elif macd_hist < 0:
            bearish_score += 10
            technical_reasons.append("MACD negative momentum acceleration")

        # Aroon Oscillator
        if aroon_osc >= 30.0:
            bullish_score += 15
            technical_reasons.append(f"Aroon Oscillator strong bullish dominance (+{aroon_osc:.0f})")
        elif aroon_osc <= -30.0:
            bearish_score += 15
            technical_reasons.append(f"Aroon Oscillator seller dominance ({aroon_osc:.0f})")

        # Relative Volume Confirmation (9-day SMA)
        rvol_9 = float(last.get('rvol_9', 1.0))
        vol_confirmed_2x = rvol_9 >= 2.0
        if vol_confirmed_2x:
            if close >= float(prev['close']):
                bullish_score += 15
                technical_reasons.append(f"Institutional Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x)")
            else:
                bearish_score += 15
                technical_reasons.append(f"Heavy Selloff Volume Confirmation (>2.0x 9-day avg: {rvol_9:.1f}x)")
        elif rvol_9 >= 1.4:
            if close >= float(prev['close']):
                bullish_score += 10
                technical_reasons.append(f"Above-average turnover ({rvol_9:.1f}x 9-day avg)")
            else:
                bearish_score += 10
                technical_reasons.append(f"Elevated distribution volume ({rvol_9:.1f}x 9-day avg)")

        # Bollinger Band squeeze
        if is_squeezing:
            technical_reasons.append("Bollinger Band volatility compression (Explosive breakout pending)")

        # Determine Signal
        if bullish_score >= 60 and bullish_score > bearish_score + 20:
            signal = "BUY"
            signal_badge = "🟢 BUY (LONG FUTURES)"
            action = "BUY_FUTURES"
            bias = "BULLISH"
            conviction = "HIGH" if bullish_score >= 80 else "MEDIUM"
            entry_min = round(max(close - (0.3 * atr), ema9), 1)
            entry_max = round(close, 1)
            entry_desc = f"₹{entry_min:,.1f} - ₹{entry_max:,.1f}"
            stop_loss = round(min(close - (1.2 * atr), ema20 * 0.995), 1)
            risk_pts = close - stop_loss
            target_1 = round(close + (1.5 * atr), 1)
            target_2 = round(close + (2.6 * atr), 1)
            reward_pts = target_1 - close
            rr_ratio = f"1:{reward_pts / (risk_pts + 1e-6):.1f}"

        elif bearish_score >= 60 and bearish_score > bullish_score + 20:
            signal = "SELL"
            signal_badge = "🔴 SELL (SHORT FUTURES)"
            action = "SELL_FUTURES"
            bias = "BEARISH"
            conviction = "HIGH" if bearish_score >= 80 else "MEDIUM"
            entry_min = round(close, 1)
            entry_max = round(min(close + (0.3 * atr), ema9), 1)
            entry_desc = f"₹{entry_min:,.1f} - ₹{entry_max:,.1f}"
            stop_loss = round(max(close + (1.2 * atr), ema20 * 1.005), 1)
            risk_pts = stop_loss - close
            target_1 = round(close - (1.5 * atr), 1)
            target_2 = round(close - (2.6 * atr), 1)
            reward_pts = close - target_1
            rr_ratio = f"1:{reward_pts / (risk_pts + 1e-6):.1f}"

        else:
            signal = "HOLD"
            signal_badge = "🟡 HOLD / RANGE-BOUND"
            action = "HOLD_NEUTRAL"
            bias = "NEUTRAL"
            conviction = "NEUTRAL"
            entry_desc = f"Consolidating around ₹{close:,.1f}"
            stop_loss = round(bb_lower, 1)
            target_1 = round(bb_upper, 1)
            target_2 = round(bb_upper + (0.5 * atr), 1)
            risk_pts = close - stop_loss
            reward_pts = target_1 - close
            rr_ratio = "1:1.0 (Non-Directional)"
            technical_reasons.append("Choppy price action within Bollinger Bands. Avoid directional futures bets.")

        # ------------------------------------------------------------------
        # 2. TWO-WAY ACTION LEVELS (BUY ABOVE / DOWNSIDE POSSIBLE BELOW)
        # ------------------------------------------------------------------
        # Classical Floor Pivots from Session High, Low, and Close
        high = float(last['high'])
        low = float(last['low'])
        pivot = (high + low + close) / 3.0
        r1 = (2.0 * pivot) - low
        s1 = (2.0 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)

        step = specs.get("strike_step", 50)
        lot = specs.get("lot_size", 65)

        # 1. Upside Breakout Trigger ("BUY ABOVE")
        # Triggers when price sustains above session high / R1 resistance
        buy_above = round(max(high, r1 * 0.999), 1)
        upside_t1 = round(buy_above + (0.75 * atr), 1)
        upside_t2 = round(buy_above + (1.5 * atr), 1)
        upside_t3 = round(buy_above + (2.5 * atr), 1)
        upside_sl = round(buy_above - (0.5 * atr), 1)
        upside_risk_pts = round(buy_above - upside_sl, 1)
        upside_reward_pts = round(upside_t1 - buy_above, 1)
        upside_rr = f"1:{upside_reward_pts / (upside_risk_pts + 1e-6):.1f}"

        # 2. Downside Breakdown Trigger ("DOWNSIDE POSSIBLE BELOW")
        # Triggers when price breaks below session low / S1 support floor
        sell_below = round(min(low, s1 * 1.001), 1)
        downside_t1 = round(sell_below - (0.75 * atr), 1)
        downside_t2 = round(sell_below - (1.5 * atr), 1)
        downside_t3 = round(sell_below - (2.5 * atr), 1)
        downside_sl = round(sell_below + (0.5 * atr), 1)
        downside_risk_pts = round(downside_sl - sell_below, 1)
        downside_reward_pts = round(sell_below - downside_t1, 1)
        downside_rr = f"1:{downside_reward_pts / (downside_risk_pts + 1e-6):.1f}"

        # 3. Consolidation / No-Trade Range
        chop_zone = f"₹{sell_below:,.1f} - ₹{buy_above:,.1f}"

        # Option plays for upside vs downside triggers
        atm_strike = int(round(close / step) * step)
        upside_opt = f"{specs['display_name']} {atm_strike if close < buy_above else atm_strike + step} CE"
        downside_opt = f"{specs['display_name']} {atm_strike if close > sell_below else atm_strike - step} PE"

        # Explicit Executive Recommendation
        if signal == "BUY":
            trade_thesis = (
                f"🟢 BULLISH TREND: Buy on dips toward ₹{entry_min:,.1f} or momentum breakout BUY ABOVE ₹{buy_above:,.1f} "
                f"for Targets ₹{upside_t1:,.1f} / ₹{upside_t2:,.1f} (SL: ₹{upside_sl:,.1f}). "
                f"Downside risk triggers strictly BELOW ₹{sell_below:,.1f} (Targets: ₹{downside_t1:,.1f} / ₹{downside_t2:,.1f})."
            )
        elif signal == "SELL":
            trade_thesis = (
                f"🔴 BEARISH TREND: Sell on pullbacks or breakdown SHORT BELOW ₹{sell_below:,.1f} "
                f"for Targets ₹{downside_t1:,.1f} / ₹{downside_t2:,.1f} (SL: ₹{downside_sl:,.1f}). "
                f"Upside invalidation occurs on sustained breakout ABOVE ₹{buy_above:,.1f} (Targets: ₹{upside_t1:,.1f} / ₹{upside_t2:,.1f})."
            )
        else:
            trade_thesis = (
                f"🟡 RANGE-BOUND ACCUMULATION: Oscillating inside chop zone ({chop_zone}). "
                f"Trade directional breakout: BUY ABOVE ₹{buy_above:,.1f} (Targets: ₹{upside_t1:,.1f} / ₹{upside_t2:,.1f}) "
                f"or breakdown SHORT BELOW ₹{sell_below:,.1f} (Targets: ₹{downside_t1:,.1f} / ₹{downside_t2:,.1f}). Avoid naked options inside the chop zone."
            )

        # ------------------------------------------------------------------
        # 3. OPTIONS EXPLORATION (STRIKES, SPREADS & STRATEGIES)
        # ------------------------------------------------------------------
        itm_strike = atm_strike - step if bias == "BULLISH" else atm_strike + step
        otm_strike = atm_strike + step if bias == "BULLISH" else atm_strike - step
        far_otm_strike = atm_strike + (2 * step) if bias == "BULLISH" else atm_strike - (2 * step)

        # Expected premium movement estimation based on Delta ~ 0.50 for ATM
        est_option_target_pts = round(reward_pts * 0.50, 1) if signal != "HOLD" else round(step * 0.6, 1)
        est_option_sl_pts = round(risk_pts * 0.45, 1) if signal != "HOLD" else round(step * 0.4, 1)
        est_profit_per_lot = round(est_option_target_pts * lot, 0)
        est_risk_per_lot = round(est_option_sl_pts * lot, 0)

        options_strategies: List[Dict[str, Any]] = []

        if signal == "BUY":
            options_strategies.append({
                "name": "Directional Call (Long ATM CE)",
                "type": "MOMENTUM_BUY",
                "recommended_contract": f"{specs['display_name']} {atm_strike} CE",
                "rationale": f"High delta (0.50) participation capturing index upside toward {target_1:,.0f}.",
                "target_points": f"+{est_option_target_pts} pts",
                "stop_loss_points": f"-{est_option_sl_pts} pts",
                "est_profit_lot": f"₹{est_profit_per_lot:,.0f}",
                "est_risk_lot": f"₹{est_risk_per_lot:,.0f}",
                "suitability": "Aggressive Traders"
            })
            options_strategies.append({
                "name": "Bull Call Spread (Hedged)",
                "type": "HEDGED_SPREAD",
                "recommended_contract": f"Buy {atm_strike} CE + Sell {otm_strike} CE",
                "rationale": "Cuts theta decay risk by 45% while maintaining favorable 1:2.0+ risk-reward.",
                "target_points": f"+{round(step * 0.65, 1)} pts net",
                "stop_loss_points": f"-{round(step * 0.35, 1)} pts net",
                "est_profit_lot": f"₹{round(step * 0.65 * lot, 0):,.0f}",
                "est_risk_lot": f"₹{round(step * 0.35 * lot, 0):,.0f}",
                "suitability": "Conservative & Capital Protection"
            })

        elif signal == "SELL":
            options_strategies.append({
                "name": "Directional Put (Long ATM PE)",
                "type": "MOMENTUM_BUY",
                "recommended_contract": f"{specs['display_name']} {atm_strike} PE",
                "rationale": f"Downside momentum play profiting from continuation toward {target_1:,.0f}.",
                "target_points": f"+{est_option_target_pts} pts",
                "stop_loss_points": f"-{est_option_sl_pts} pts",
                "est_profit_lot": f"₹{est_profit_per_lot:,.0f}",
                "est_risk_lot": f"₹{est_risk_per_lot:,.0f}",
                "suitability": "Aggressive Short Sellers"
            })
            options_strategies.append({
                "name": "Bear Put Spread (Hedged)",
                "type": "HEDGED_SPREAD",
                "recommended_contract": f"Buy {atm_strike} PE + Sell {otm_strike} PE",
                "rationale": "Defined risk spread limiting premium loss if market encounters a sudden short covering rally.",
                "target_points": f"+{round(step * 0.65, 1)} pts net",
                "stop_loss_points": f"-{round(step * 0.35, 1)} pts net",
                "est_profit_lot": f"₹{round(step * 0.65 * lot, 0):,.0f}",
                "est_risk_lot": f"₹{round(step * 0.35 * lot, 0):,.0f}",
                "suitability": "Conservative Traders"
            })

        else:
            # HOLD / Range-bound
            if is_squeezing:
                options_strategies.append({
                    "name": "Long Volatility Straddle",
                    "type": "VOLATILITY_BREAKOUT",
                    "recommended_contract": f"Buy {atm_strike} CE + Buy {atm_strike} PE",
                    "rationale": "Bollinger squeeze indicates imminent multi-hundred point explosion in either direction.",
                    "target_points": f"+{round(atr * 0.8, 1)} pts breakout",
                    "stop_loss_points": f"-{round(atr * 0.35, 1)} pts theta",
                    "est_profit_lot": f"₹{round(atr * 0.8 * lot, 0):,.0f}",
                    "est_risk_lot": f"₹{round(atr * 0.35 * lot, 0):,.0f}",
                    "suitability": "Breakout Traders"
                })
            else:
                options_strategies.append({
                    "name": "Iron Condor / Short Strangle",
                    "type": "THETA_INCOME",
                    "recommended_contract": f"Sell {otm_strike} CE & Sell {itm_strike} PE",
                    "rationale": "Sideways market with high probability of expiration between Bollinger bands.",
                    "target_points": f"+{round(step * 0.4, 1)} pts theta decay",
                    "stop_loss_points": f"-{round(step * 0.7, 1)} pts",
                    "est_profit_lot": f"₹{round(step * 0.4 * lot, 0):,.0f}",
                    "est_risk_lot": f"₹{round(step * 0.7 * lot, 0):,.0f}",
                    "suitability": "Option Sellers (High Margin)"
                })

        # ------------------------------------------------------------------
        # 3. PROFITABILITY SCORING & RANKING
        # ------------------------------------------------------------------
        # Score from 0 to 100 assessing how profitable and high-conviction this index is
        profitability_score = 40.0

        if signal in ["BUY", "SELL"]:
            profitability_score += 25.0
            if conviction == "HIGH":
                profitability_score += 15.0

            # Reward points relative to index scale
            atr_pct = (atr / close) * 100.0
            if atr_pct >= 0.9:
                profitability_score += 10.0  # High volatility gives huge point movement
            
            if is_squeezing:
                profitability_score += 10.0  # Volatility expansion ready

        profitability_score = min(round(profitability_score, 1), 100.0)

        return {
            "symbol": clean_sym,
            "display_name": specs.get("display_name", clean_sym),
            "exchange": specs.get("exchange", "NSE"),
            "description": specs.get("description", ""),
            "close": round(close, 2),
            "prev_close": round(prev_close, 2),
            "change_pts": round(change_pts, 2),
            "change_pct": round(change_pct, 2),
            "lot_size": lot,
            "strike_step": step,
            # Futures Signal
            "signal": signal,
            "signal_badge": signal_badge,
            "action": action,
            "bias": bias,
            "conviction": conviction,
            "futures_plan": {
                "entry_zone": entry_desc,
                "stop_loss": stop_loss,
                "target_1": target_1,
                "target_2": target_2,
                "risk_pts": round(risk_pts, 1),
                "reward_pts": round(reward_pts, 1),
                "risk_reward_ratio": rr_ratio,
                "point_value_per_lot": f"₹{lot:,.0f} per 1 index point",
                "max_profit_target_1": f"₹{round(reward_pts * lot, 0):,.0f} / lot"
            },
            # Two-Way Action Levels (Buy Above / Downside Possible Below)
            "action_levels": {
                "buy_above": buy_above,
                "upside_target_1": upside_t1,
                "upside_target_2": upside_t2,
                "upside_target_3": upside_t3,
                "upside_stop_loss": upside_sl,
                "upside_risk_pts": upside_risk_pts,
                "upside_reward_pts": upside_reward_pts,
                "upside_risk_reward": upside_rr,
                "upside_option_play": upside_opt,
                "sell_below": sell_below,
                "downside_target_1": downside_t1,
                "downside_target_2": downside_t2,
                "downside_target_3": downside_t3,
                "downside_stop_loss": downside_sl,
                "downside_risk_pts": downside_risk_pts,
                "downside_reward_pts": downside_reward_pts,
                "downside_risk_reward": downside_rr,
                "downside_option_play": downside_opt,
                "chop_zone": chop_zone,
                "trade_thesis": trade_thesis,
                "pivots": {
                    "pivot": round(pivot, 1),
                    "r1": round(r1, 1),
                    "r2": round(r2, 1),
                    "s1": round(s1, 1),
                    "s2": round(s2, 1)
                }
            },
            # Options Setup
            "options_setup": {
                "atm_strike": atm_strike,
                "itm_strike": itm_strike,
                "otm_strike": otm_strike,
                "strategies": options_strategies
            },
            # Indicators
            "indicators": {
                "rsi_14": round(rsi, 1),
                "ema_20": round(ema20, 1),
                "ema_50": round(ema50, 1),
                "sma_200": round(sma200, 1),
                "atr_14": round(atr, 1),
                "macd_line": round(macd_line, 2),
                "macd_signal": round(macd_signal, 2),
                "macd_hist": round(macd_hist, 2),
                "macd_level2": macd_level2,
                "aroon_osc": round(aroon_osc, 1),
                "aroon_up": round(aroon_up, 1),
                "aroon_down": round(aroon_down, 1),
                "bb_bandwidth": round(bbw, 4),
                "is_squeezing": is_squeezing,
                "clv": round(clv, 3),
                "rvol_9": round(rvol_9, 2),
                "vol_confirmed_2x": vol_confirmed_2x
            },
            "technical_reasons": technical_reasons,
            "profitability_score": profitability_score
        }

    @staticmethod
    def analyze_batch(data_dict: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Processes all loaded index DataFrames, generates trading plans,
        and returns them sorted by profitability score descending.
        """
        results = []
        for symbol, df in data_dict.items():
            try:
                res = IndexDerivativesAnalyzer.analyze_index(symbol, df)
                if res:
                    results.append(res)
            except Exception:
                continue

        # Sort by highest profitability score descending
        results.sort(key=lambda x: x.get("profitability_score", 0), reverse=True)
        return results
