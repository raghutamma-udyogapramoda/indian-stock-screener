"""
Streamlit Web Dashboard for Indian Equities Screener & AI Trade Advisor.
Supports: BTST, Breakout, Breakdown, Swing, Intraday, and Options (CE/PE).
Run with: streamlit run ui/app.py
"""

import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure proper Unicode/emoji handling on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ai.advisor import AIAdvisor
from core.auth import render_login_gate, render_sidebar_user_badge, get_current_user
from core.indices import IndexDerivativesAnalyzer, INDEX_SPECS
from core.tracker import PortfolioTracker
from core.universe import UniverseManager
from core.indicators import enrich_with_indicators
from providers.breeze_provider import BreezeProvider
from providers.breeze_static_provider import BreezeStaticProvider
from providers.nse_direct_provider import NSEDirectProvider
from providers.yfinance_provider import YahooFinanceProvider
from screeners.breakdown import BreakdownScreener
from screeners.breakout import BreakoutScreener
from screeners.btst import BTSTScreener
from screeners.intraday import IntradayScreener
from screeners.options_fno import OptionsScreener
from screeners.swing import SwingScreener

st.set_page_config(
    page_title="AI Stock & Commodity Screener | NSE, BSE & MCX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str):
    """Safely renders HTML without markdown parser converting indented tags into code blocks."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)


def plot_position_chart(symbol: str, df: pd.DataFrame, buy_price: float, stop_loss: float, target: float):
    """Renders a candlestick chart highlighting entry price, stop-loss, and target levels."""
    recent = df.tail(60).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['open'],
        high=recent['high'],
        low=recent['low'],
        close=recent['close'],
        name="Price",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF5252"
    ))
    fig.add_hline(y=buy_price, line_dash="dash", line_color="#2962FF", annotation_text=f"Buy: ₹{buy_price:,.2f}", annotation_position="top left")
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="#FF5252", annotation_text=f"SL: ₹{stop_loss:,.2f}", annotation_position="bottom left")
    fig.add_hline(y=target, line_dash="dashdot", line_color="#00E676", annotation_text=f"Target: ₹{target:,.2f}", annotation_position="top right")
    
    fig.update_layout(
        title=f"{symbol} — Trailing Position Technicals & Trigger Levels",
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)


def render_item_tracker_view(current_user: str, data_dict: Dict[str, pd.DataFrame]):
    """Renders the comprehensive Live Item Tracker for the authenticated user."""
    positions = PortfolioTracker.load_positions(current_user)

    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.subheader(f"📌 {current_user.upper()}'s Live Item Tracker & Portfolio Watch")
        st.caption("Real-time position monitoring with intelligent BUY, HOLD, EXIT, and AVERAGE DOWN (Fake Fall / Shakeout Detection) signals.")
    with h_col2:
        if positions:
            if st.button("🧹 Clear All Positions", type="secondary", use_container_width=True, key=f"clear_all_{current_user}", help="Remove all tracked items"):
                PortfolioTracker.clear_all_positions(current_user)
                st.toast("✅ Cleared all tracked positions.", icon="🧹")
                time.sleep(0.3)
                st.rerun()

    # 1. Fetch live data for any tracked positions not in data_dict
    needed_syms = []
    for p in positions:
        und = p.get("underlying") or PortfolioTracker.get_underlying_symbol(p.get("symbol", ""))
        if und and und not in data_dict:
            needed_syms.append(und)
    
    active_data = dict(data_dict)
    if needed_syms:
        try:
            yf_fallback = YahooFinanceProvider(cache_ttl_hours=1.0)
            fetched = yf_fallback.fetch_batch_ohlcv(list(set(needed_syms)), period="6mo", interval="1d", max_workers=5)
            for s, df in fetched.items():
                if df is not None and not df.empty:
                    active_data[s] = df
        except Exception:
            pass

    # 2. Evaluate all positions
    evaluated = []
    for p in positions:
        und = p.get("underlying") or PortfolioTracker.get_underlying_symbol(p.get("symbol", ""))
        df_for_pos = active_data.get(und)
        if df_for_pos is None or (isinstance(df_for_pos, pd.DataFrame) and df_for_pos.empty):
            df_for_pos = active_data.get(p.get("symbol", ""))
        res = PortfolioTracker.evaluate_live_position(p, df_for_pos)
        evaluated.append(res)

    # 3. Portfolio Summary KPIs
    tot_invested = sum(r["invested_val"] for r in evaluated) if evaluated else 0.0
    tot_current = sum(r["current_val"] for r in evaluated) if evaluated else 0.0
    tot_pnl = tot_current - tot_invested
    tot_pnl_pct = (tot_pnl / tot_invested * 100) if tot_invested > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Portfolio Value", f"₹{tot_current:,.2f}")
    kpi2.metric("Total Invested Capital", f"₹{tot_invested:,.2f}")
    kpi3.metric(
        "Total Unrealized P&L",
        f"₹{tot_pnl:+,.2f} ({tot_pnl_pct:+.2f}%)",
        delta=f"{tot_pnl_pct:+.2f}%" if tot_invested > 0 else "0.00%",
        delta_color="normal" if tot_invested > 0 else "off"
    )
    kpi4.metric("Active Positions", f"{len(positions)} Tracked")

    # Action distribution summary
    if evaluated:
        n_avg = sum(1 for r in evaluated if r["action_type"] == "AVERAGE_DOWN")
        n_hold = sum(1 for r in evaluated if "HOLD" in r["action_type"])
        n_fake_rise = sum(1 for r in evaluated if r["action_type"] == "FAKE_RISE")
        n_sl = sum(1 for r in evaluated if r["action_type"] == "EXIT")
        n_tgt = sum(1 for r in evaluated if r["action_type"] == "TARGET_HIT")

        st.markdown(
            f'<div style="background: #131722; border: 1px solid #2a2e39; border-radius: 8px; padding: 10px 16px; margin: 12px 0 20px 0; font-size: 0.85rem;">'
            f'<b>Decision Distribution:</b> '
            f'<span style="color: #00E676; margin-right: 14px;">🟢 Average Down (Fake Fall): <b>{n_avg}</b></span> '
            f'<span style="color: #64B5F6; margin-right: 14px;">🟢 Strong Hold: <b>{n_hold}</b></span> '
            f'<span style="color: #FF9800; margin-right: 14px;">⚠️ Fake Rise Alert: <b>{n_fake_rise}</b></span> '
            f'<span style="color: #FF5252; margin-right: 14px;">🛑 Exit (SL Broken): <b>{n_sl}</b></span> '
            f'<span style="color: #FFD54F;">🎯 Target Hit: <b>{n_tgt}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 4. Form: Add Position Manually
    with st.expander("➕ Add Stock / Index / Option / Future to Tracker", expanded=(len(positions) == 0)):
        with st.form("manual_add_tracker_form", clear_on_submit=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            new_sym = f_col1.text_input("Asset Symbol / Contract", placeholder="e.g. RELIANCE, NIFTY 25000 CE, GOLD")
            new_type = f_col2.selectbox("Asset Type", ["EQUITY", "INDEX", "FUTURES", "OPTION_CE", "OPTION_PE", "COMMODITY"])
            new_buy = f_col3.number_input("Your Buy / Entry Price (₹)", min_value=0.0, step=1.0, value=0.0, help="Enter the exact purchase price at which you bought this asset")

            f_col4, f_col5, f_col6 = st.columns(3)
            new_qty = f_col4.number_input("Quantity / Units / Lots", min_value=1.0, step=1.0, value=1.0)
            new_sl = f_col5.number_input("Stop Loss (₹, optional)", min_value=0.0, step=1.0, value=0.0)
            new_tgt = f_col6.number_input("Target Price (₹, optional)", min_value=0.0, step=1.0, value=0.0)

            new_notes = st.text_input("Strategy Notes (Optional)", placeholder="e.g. Breakout setup from screener, monthly expiry hedge")

            add_submit = st.form_submit_button("➕ Add to My Live Tracker", type="primary", use_container_width=True)
            if add_submit:
                if not new_sym.strip():
                    st.error("Please provide a valid symbol.")
                elif float(new_buy) <= 0:
                    st.error("Please enter your actual Buy / Entry Price (greater than ₹0).")
                else:
                    PortfolioTracker.add_position(
                        username=current_user,
                        symbol=new_sym.strip(),
                        buy_price=float(new_buy),
                        qty=float(new_qty),
                        asset_type=new_type,
                        stop_loss=float(new_sl) if new_sl > 0 else None,
                        target=float(new_tgt) if new_tgt > 0 else None,
                        notes=new_notes
                    )
                    st.toast(f"✅ Added {new_sym.upper()} to your tracker!", icon="📌")
                    time.sleep(0.4)
                    st.rerun()

    # 5. Position Cards
    if not evaluated:
        st.info("💡 You have no tracked positions yet. Use the form above to add stocks/options you bought, or click **'📌 Track this Position'** on any setup in the **AI Trade Plans** tab.")
    else:
        st.markdown("### 📋 Active Tracked Positions & Live Decision Engine")
        for idx, item in enumerate(evaluated):
            pos_id = item["id"]
            action_type = item["action_type"]
            badge_color = item["action_color"]
            border_color = badge_color
            bg_color = "#131722"

            pnl_val = item["pnl_val"]
            pnl_pct = item["pnl_pct"]
            pnl_badge_color = "#00E676" if pnl_val >= 0 else "#FF5252"
            pnl_badge_bg = "rgba(0, 230, 118, 0.15)" if pnl_val >= 0 else "rgba(255, 82, 82, 0.15)"

            # Callout card
            card_html = (
                f'<div style="background-color: {bg_color}; border-left: 6px solid {border_color}; border-radius: 10px; padding: 18px; margin-bottom: 12px; border-top: 1px solid #2a2e39; border-right: 1px solid #2a2e39; border-bottom: 1px solid #2a2e39; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
                f'<div>'
                f'<span style="background-color: #2962FF; color: #FFF; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">{item["asset_type"]}</span>'
                f'<span style="font-size: 1.35rem; font-weight: bold; color: #FFF;">{item["symbol"]}</span> '
                f'<span style="color: #787b86; font-size: 0.85rem; margin-left: 8px;">Bought: {item["buy_date"]}</span>'
                f'</div>'
                f'<div style="text-align: right;">'
                f'<span style="background-color: {badge_color}; color: #000; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85rem;">{item["action_badge"]}</span>'
                f'</div>'
                f'</div>'
                f'<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; background-color: #0e1117; padding: 12px; border-radius: 6px; margin: 14px 0 10px 0;">'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Buy Price</span><br><b style="color: #FFF; font-size: 1.05rem;">₹{item["buy_price"]:,.2f}</b></div>'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Live CMP</span><br><b style="color: #00E676; font-size: 1.05rem;">₹{item["current_price"]:,.2f}</b></div>'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Quantity</span><br><b style="color: #FFF; font-size: 1.05rem;">{item["qty"]}</b></div>'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Unrealized P&L</span><br><b style="color: {pnl_badge_color}; font-size: 1.05rem; background: {pnl_badge_bg}; padding: 2px 6px; border-radius: 4px;">{pnl_pct:+.2f}% (₹{pnl_val:+,.2f})</b></div>'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Stop Loss</span><br><b style="color: #FF5252; font-size: 1.05rem;">₹{item["stop_loss"]:,.2f}</b></div>'
                f'<div><span style="color: #787b86; font-size: 0.75rem;">Target</span><br><b style="color: #64B5F6; font-size: 1.05rem;">₹{item["target"]:,.2f}</b></div>'
                f'</div>'
                f'<div style="background-color: #1a1e29; border-left: 4px solid {border_color}; border-radius: 6px; padding: 10px 14px; margin-top: 10px;">'
                f'<div style="font-size: 0.88rem; color: #FFF; margin-bottom: 4px;"><b>🧠 Decision Rationale:</b> {item["rationale"]}</div>'
                f'<div style="font-size: 0.88rem; color: #FFD54F;"><b>💡 Suggested Tactical Action:</b> {item["suggested_action"]}</div>'
                f'</div>'
                f'</div>'
            )
            render_html(card_html)

            # Chart and Deletion expander
            und = item.get("underlying") or PortfolioTracker.get_underlying_symbol(item["symbol"])
            df_for_pos = active_data.get(und)
            if df_for_pos is None or (isinstance(df_for_pos, pd.DataFrame) and df_for_pos.empty):
                df_for_pos = active_data.get(item["symbol"])
            with st.expander(f"📊 Chart & Position Controls for {item['symbol']}"):
                if df_for_pos is not None and not df_for_pos.empty:
                    plot_position_chart(item['symbol'], df_for_pos, item['buy_price'], item['stop_loss'], item['target'])
                
                col_del, col_space = st.columns([1, 4])
                with col_del:
                    if st.button("🗑️ Delete Position", key=f"del_pos_{pos_id}_{idx}", type="secondary"):
                        PortfolioTracker.delete_position(current_user, pos_id)
                        st.toast(f"Removed {item['symbol']} from your tracker.", icon="🗑️")
                        time.sleep(0.3)
                        st.rerun()


# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1e24;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 5px solid #2962FF;
    }
    .badge-buy {
        background-color: #00E676;
        color: #000;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-sell {
        background-color: #FF5252;
        color: #FFF;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- AUTHENTICATION GATE -----------------
# Ensure only authenticated users can access screeners and market results
if not render_login_gate():
    st.stop()

# ----------------- SIDEBAR -----------------
render_sidebar_user_badge()
st.sidebar.title("⚡ Screener Controls")

provider_choice = st.sidebar.selectbox(
    "Data Feed Provider",
    [
        "Yahoo Finance (Default)",
        "NSE Direct (Official Exchange)",
        "ICICI Breeze (Static / Test)",
        "ICICI Breeze (Live Account)"
    ],
    index=0,
    help="Yahoo Finance: Real-time global feeds. NSE Direct: Authoritative official National Stock Exchange Bhavcopy. ICICI Breeze: Broker account."
)

breeze_api_key_in = ""
breeze_secret_key_in = ""
breeze_session_token_in = ""

# Auto-detect session token if redirected from Breeze API login URL
url_session_token = ""
if hasattr(st, "query_params") and "apisession" in st.query_params:
    url_session_token = st.query_params.get("apisession", "")

if provider_choice == "ICICI Breeze (Live Account)":
    with st.sidebar.expander("🔑 Breeze Broker Credentials", expanded=True):
        breeze_api_key_in = st.text_input(
            "Breeze App Key",
            value=os.getenv("BREEZE_API_KEY", ""),
            type="password",
            help="Found under App Settings on https://api.icicidirect.com"
        )
        breeze_secret_key_in = st.text_input(
            "Breeze Secret Key",
            value=os.getenv("BREEZE_SECRET_KEY", ""),
            type="password"
        )
        
        # Pre-fill with auto-captured URL session token if present
        default_token = url_session_token or os.getenv("BREEZE_SESSION_TOKEN", "")
        if url_session_token:
            st.success("✅ Auto-captured Session Token from URL!")
            
        breeze_session_token_in = st.text_input(
            "Daily Session Token",
            value=default_token,
            type="password",
            help="Generate daily by logging into the ICICI Direct API portal."
        )

        if breeze_api_key_in and not breeze_api_key_in.startswith("your_"):
            login_url = f"https://api.icicidirect.com/apiuser/login?api_key={breeze_api_key_in}"
            st.markdown(f'<a href="{login_url}" target="_blank" style="display: block; text-align: center; background: #004D25; color: #00E676; border: 1px solid #00E676; border-radius: 4px; padding: 6px; font-size: 0.82rem; font-weight: bold; text-decoration: none; margin-top: 6px;">🔗 Login to Breeze (Get Daily Token)</a>', unsafe_allow_html=True)
            st.caption("Redirects back to this app with token auto-filled.")

universe_choice = st.sidebar.selectbox(
    "Select Asset Universe",
    ["NIFTY_50", "🏛️ INDICES (NIFTY, BankNifty, Sensex...)", "NSE_FO", "NIFTY_500", "MCX Commodities", "Custom"],
    index=0,
    help="NIFTY_50: Top 50 bluechips. INDICES: Nifty, BankNifty, Sensex, Midcap, Sectoral indices. NSE_FO: 180+ liquid F&O stocks. MCX Commodities: Gold, Silver, Crude, Gas, Metals. NIFTY_500: Broad market."
)

custom_tickers = ""
if universe_choice == "Custom":
    custom_tickers = st.sidebar.text_area(
        "Enter Tickers (comma separated)",
        "RELIANCE, TCS, INFY, TMPV, HDFCBANK, AXISBANK, TRENT, BEL, GOLD, CRUDEOIL"
    )

strategy_choice = st.sidebar.selectbox(
    "Strategy Filter",
    [
        "All Strategies",
        "Breakout (Multi-Week & 52W)",
        "Breakdown (Short & Distribution)",
        "Swing (Pullback & Value)",
        "BTST (Overnight Momentum)",
        "Intraday Momentum",
        "Options (CE & PE Buildup)",
        "Options CE (Call Options Only)",
        "Options PE (Put Options Only)"
    ],
    index=0
)

top_n = st.sidebar.slider("Top Candidates per Category", min_value=3, max_value=15, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Advisor Settings")
gemini_key_input = st.sidebar.text_input(
    "Gemini API Key (Optional)",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Get a free key from https://aistudio.google.com. If omitted, built-in Mathematical Engine generates trade plans."
)

force_live_refresh = st.sidebar.checkbox(
    "⚡ Live Force Refresh (Bypass Cache)",
    value=False,
    help="Forces immediate fresh downloads from exchange servers, bypassing local disk cache."
)

cache_ttl = st.sidebar.slider("Cache Freshness (Hours)", min_value=1, max_value=24, value=4)
run_btn = st.sidebar.button("🔍 Run Live Market Screen", type="primary", use_container_width=True)

if st.sidebar.button("🧹 Flush Cache & Reset Memory", use_container_width=True):
    st.cache_data.clear()
    st.session_state.clear()
    st.toast("✅ App memory cache flushed! Re-running screeners...", icon="🧹")
    st.rerun()

# ----------------- MAIN APP -----------------
st.title("📈 Indian Equities & Commodities AI Screener")
st.caption("Quantitative Multi-Strategy Algorithmic Screener with Low-Cost AI Trade Plan Generation (NSE, BSE & MCX)")

# Helper to run scan (disk-cached by LocalDataCache)
def execute_screening(u_name, custom_list, strat, top_limit, prov_mode, b_key="", b_sec="", b_tok="", bypass_cache=False):
    t_start = time.time()
    
    # 1. Load symbols
    if u_name == "Custom":
        symbols = [s.strip().upper() for s in custom_list.replace(",", " ").split() if s.strip()]
    elif u_name == "MCX Commodities":
        symbols = UniverseManager.get_tickers("MCX")
    elif "INDICES" in u_name.upper():
        symbols = UniverseManager.get_tickers("INDICES")
    else:
        symbols = UniverseManager.get_tickers(u_name)

    # 2. Select data provider
    if "Static" in prov_mode:
        provider = BreezeStaticProvider()
        data = provider.fetch_batch_ohlcv(symbols, period="1y", interval="1day", max_workers=5)
    elif "Live" in prov_mode:
        provider = BreezeProvider(api_key=b_key, secret_key=b_sec, session_token=b_tok)
        if not provider.is_authenticated:
            st.warning("⚠️ Live ICICI Breeze session not active or credentials missing. Using Breeze Static Provider for testing.")
            provider = BreezeStaticProvider()
        data = provider.fetch_batch_ohlcv(symbols, period="1y", interval="1day", max_workers=5)
    elif "NSE Direct" in prov_mode:
        provider = NSEDirectProvider(cache_ttl_hours=cache_ttl)
        data = provider.fetch_batch_ohlcv(symbols, period="1y", interval="1d", max_workers=15, use_cache=not bypass_cache)
    else:
        provider = YahooFinanceProvider(cache_ttl_hours=cache_ttl)
        data = provider.fetch_batch_ohlcv(symbols, period="1y", interval="1d", max_workers=15, use_cache=not bypass_cache)
    
    # 3. Always compute Indian Market Indices Derivatives Hub
    idx_symbols = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT"]
    yf_idx_prov = YahooFinanceProvider(cache_ttl_hours=cache_ttl)
    idx_raw = yf_idx_prov.fetch_batch_ohlcv(idx_symbols, period="6mo", interval="1d", max_workers=5, use_cache=not bypass_cache)
    index_results = IndexDerivativesAnalyzer.analyze_batch(idx_raw)

    # 4. Extract data health telemetry from providers
    p_health = provider.get_health_status() if hasattr(provider, "get_health_status") else {}
    idx_health = yf_idx_prov.get_health_status() if hasattr(yf_idx_prov, "get_health_status") else {}

    api_call_failed = p_health.get("api_call_failed", False) or idx_health.get("api_call_failed", False)
    failed_syms = sorted(list(set(p_health.get("failed_symbols", []) + idx_health.get("failed_symbols", []))))
    fallback_syms = sorted(list(set(p_health.get("fallback_symbols", []) + idx_health.get("fallback_symbols", []))))
    failure_reasons = p_health.get("failure_reasons", []) + idx_health.get("failure_reasons", [])

    # Find latest candle date across all loaded datasets (normalizing tz to avoid comparison errors)
    all_dates = []
    for df in list(data.values()) + list(idx_raw.values()):
        if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[-1]
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            all_dates.append(ts)
    latest_data_date = max(all_dates).strftime("%d-%b-%Y") if all_dates else "N/A"

    is_static = "Static" in prov_mode or p_health.get("data_source_mode") == "STATIC_TEST"
    is_latest = (not api_call_failed) and (not is_static) and p_health.get("is_latest", True) and (len(data) > 0)

    data_health = {
        "is_latest": is_latest,
        "api_call_failed": api_call_failed,
        "provider_name": prov_mode,
        "data_source_mode": p_health.get("data_source_mode", "LIVE"),
        "failed_symbols": failed_syms,
        "fallback_symbols": fallback_syms,
        "failure_reasons": failure_reasons,
        "latest_data_date": latest_data_date,
        "total_symbols_requested": len(symbols),
        "total_symbols_loaded": len(data),
        "is_static_mode": is_static
    }

    # 5. Screen Equities with Benchmark Context (Sector Tailwind & Panic-Day Behavior)
    results = {}
    if strat in ["All Strategies", "Breakout (Multi-Week & 52W)"]:
        results["BREAKOUT"] = BreakoutScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    if strat in ["All Strategies", "Breakdown (Short & Distribution)"]:
        results["BREAKDOWN"] = BreakdownScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    if strat in ["All Strategies", "Swing (Pullback & Value)"]:
        results["SWING"] = SwingScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    if strat in ["All Strategies", "BTST (Overnight Momentum)"]:
        results["BTST"] = BTSTScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    if strat in ["All Strategies", "Intraday Momentum"]:
        results["INTRADAY"] = IntradayScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    if strat in ["All Strategies", "Options (CE & PE Buildup)"]:
        results["OPTIONS"] = OptionsScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    elif strat == "Options CE (Call Options Only)":
        results["OPTIONS_CE"] = OptionsScreener(option_target="CE").screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
    elif strat == "Options PE (Put Options Only)":
        results["OPTIONS_PE"] = OptionsScreener(option_target="PE").screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)

    elapsed = time.time() - t_start
    return symbols, data, results, index_results, elapsed, data_health


if run_btn or "cached_results" in st.session_state:
    if run_btn:
        with st.spinner("Syncing candles & computing vectorized indicators..."):
            symbols, data_dict, results, index_results, elapsed, data_health = execute_screening(
                universe_choice,
                custom_tickers,
                strategy_choice,
                top_n,
                provider_choice,
                breeze_api_key_in,
                breeze_secret_key_in,
                breeze_session_token_in,
                force_live_refresh
            )
            st.session_state["symbols"] = symbols
            st.session_state["data_dict"] = data_dict
            st.session_state["cached_results"] = results
            st.session_state["index_results"] = index_results
            st.session_state["elapsed"] = elapsed
            st.session_state["data_health"] = data_health

    symbols = st.session_state["symbols"]
    data_dict = st.session_state["data_dict"]
    results = st.session_state["cached_results"]
    index_results = st.session_state.get("index_results", [])
    elapsed = st.session_state["elapsed"]
    data_health = st.session_state.get("data_health", {
        "is_latest": True,
        "api_call_failed": False,
        "provider_name": provider_choice,
        "latest_data_date": "Recent Session"
    })

    # Auto-refresh indices if cached in previous session without action_levels, missing triggers, or with obsolete lot sizes
    needs_refresh = (
        not index_results
        or any(
            not idx.get("action_levels") 
            or not idx.get("action_levels", {}).get("buy_above")
            or not idx.get("action_levels", {}).get("trade_thesis")
            or idx.get("lot_size") != INDEX_SPECS.get(idx.get("symbol"), {}).get("lot_size", 65)
            for idx in index_results
        )
    )
    if needs_refresh:
        try:
            idx_symbols = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT"]
            yf_idx_prov = YahooFinanceProvider(cache_ttl_hours=1)
            idx_raw = yf_idx_prov.fetch_batch_ohlcv(idx_symbols, period="6mo", interval="1d", max_workers=5, use_cache=True)
            index_results = IndexDerivativesAnalyzer.analyze_batch(idx_raw)
            st.session_state["index_results"] = index_results
        except Exception:
            pass

    # Gather all candidates
    all_candidates = []
    for cat, c_list in results.items():
        all_candidates.extend(c_list)

    # ----------------- PROMINENT EXTERNAL API FAILURE / DATA STALENESS HEADER -----------------
    is_live = data_health.get("is_latest", True) and not data_health.get("api_call_failed", False)
    
    if not is_live:
        prov_name = data_health.get("provider_name", provider_choice)
        latest_dt = data_health.get("latest_data_date", "Prior Session")
        failed_count = len(data_health.get("failed_symbols", []))
        
        if data_health.get("is_static_mode"):
            banner_title = "⚠️ TEST / MOCK FEED ACTIVE — DATA FETCHED IS NOT LATEST"
            banner_msg = (
                f"You are currently viewing offline mock/test fixtures for <b>{prov_name}</b>. "
                f"This data is simulated for testing and does <b>NOT</b> reflect current live exchange prices or today's market session."
            )
        elif data_health.get("api_call_failed"):
            failed_str = f" for <b>{failed_count} symbols</b>" if failed_count > 0 else ""
            banner_title = "⚠️ EXTERNAL API CALL FAILED — DATA FETCHED IS NOT LATEST"
            banner_msg = (
                f"The API call to the external market feed (<b>{prov_name}</b>) failed{failed_str} due to network failure, API timeout, or exchange rate limiting. "
                f"<br>The application has automatically fallen back to <b>cached / historical data</b> (latest session available: <b>{latest_dt}</b>). "
                f"<br><b>Important Trading Warning:</b> Current Market Prices (CMP), breakout triggers, stop losses, and indicator signals shown on this dashboard <b>do not reflect current live market prices</b>."
            )
        else:
            banner_title = "⚠️ CACHED MARKET DATA ACTIVE — DATA FETCHED IS NOT LATEST"
            banner_msg = (
                f"Data displayed was retrieved from local disk cache (Latest session: <b>{latest_dt}</b>). "
                f"To sync live prices directly from the exchange, enable <b>'⚡ Live Force Refresh (Bypass Cache)'</b> in the sidebar and rerun the screen."
            )

        warn_banner_html = (
            f'<div style="background: linear-gradient(135deg, #3d0c02 0%, #7f1d1d 50%, #450a0a 100%); '
            f'border: 2px solid #EF4444; border-radius: 10px; padding: 18px 24px; margin-top: 10px; margin-bottom: 22px; '
            f'box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);">'
            f'<div style="display: flex; align-items: flex-start; gap: 16px;">'
            f'<div style="font-size: 2.6rem; line-height: 1;">⚠️</div>'
            f'<div style="flex-grow: 1;">'
            f'<div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.5px; text-transform: uppercase;">{banner_title}</div>'
            f'<div style="font-size: 0.98rem; color: #FEE2E2; margin-top: 6px; line-height: 1.5;">{banner_msg}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        render_html(warn_banner_html)

        if data_health.get("failure_reasons") or data_health.get("failed_symbols"):
            with st.expander("🛠️ View External API Diagnostic Details & Failure Logs", expanded=False):
                if data_health.get("failure_reasons"):
                    st.write("**External System Error Messages:**")
                    for err in data_health.get("failure_reasons", [])[:10]:
                        st.code(err, language="text")
                if data_health.get("failed_symbols"):
                    st.write(f"**Failed Symbols ({len(data_health['failed_symbols'])}):** {', '.join(data_health['failed_symbols'][:30])}")
                if data_health.get("fallback_symbols"):
                    st.write(f"**Loaded from Fallback Cache ({len(data_health['fallback_symbols'])}):** {', '.join(data_health['fallback_symbols'][:30])}")
    else:
        active_banner_html = (
            f'<div style="background: linear-gradient(90deg, #064e3b 0%, #047857 100%); border: 1px solid #10B981; '
            f'border-radius: 8px; padding: 8px 16px; margin-top: 8px; margin-bottom: 18px; display: flex; '
            f'align-items: center; justify-content: space-between;">'
            f'<div style="color: #ECFDF5; font-size: 0.9rem; font-weight: 600;">'
            f'🟢 <b>Live Data Feed Active</b> — Successfully connected to {data_health.get("provider_name", provider_choice)}. Data is latest (session: {data_health.get("latest_data_date", "today")}).'
            f'</div>'
            f'<div style="color: #A7F3D0; font-size: 0.8rem;">Exchange Connection: Verified</div>'
            f'</div>'
        )
        render_html(active_banner_html)

    # Top KPI Row (5 columns)
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Universe Analyzed", f"{len(symbols)} Stocks")
    kpi2.metric("Scan Execution Time", f"{elapsed:.2f}s")
    kpi3.metric("Qualified Setups", f"{len(all_candidates)} Found")
    kpi4.metric(
        "Data Feed Health",
        "🟢 Live (Latest)" if is_live else "⚠️ Not Latest",
        delta="Connected" if is_live else "External API Error",
        delta_color="normal" if is_live else "inverse"
    )
    kpi5.metric(
        "Market Session Date",
        data_health.get("latest_data_date", "N/A"),
        delta="Live Feed" if is_live else "From Cache",
        delta_color="normal" if is_live else "off"
    )

    st.markdown("---")

    tab_indices, tab_ai, tab_screeners, tab_tracker, tab_charts, tab_docs, tab_audit = st.tabs([
        "🏛️ Indices Hub",
        "🤖 AI Trade Plans",
        "📊 Strategy Shortlists",
        "📌 Live Item Tracker",
        "📈 Technical Chart View",
        "⚙️ Strategy Documentation",
        "🔍 API Data Audit & Proof"
    ])

    with tab_indices:
        st.subheader("🏛️ Indian Market & Sectoral Indices Derivatives Hub")
        st.caption("Quantitative Futures signals (Buy / Sell / Hold), Option Strategies (CE/PE/Spreads), and Profitability Ranking for NIFTY 50, BANK NIFTY, SENSEX, MIDCAP NIFTY, and Sectoral Indices.")

        if not is_live:
            st.warning(f"⚠️ **Notice on Index Pricing:** External market feed status is **Not Latest**. All index levels, futures signals, and action triggers are calculated from cached session data ({data_health.get('latest_data_date', 'prior session')}).")

        if not index_results:
            st.warning("Index data is currently loading...")
        else:
            top_rec = index_results[0]
            banner_bg = "linear-gradient(90deg, #1e3c72 0%, #2a5298 100%)" if "BUY" in top_rec['signal'] else ("linear-gradient(90deg, #4a0e17 0%, #2b080c 100%)" if "SELL" in top_rec['signal'] else "linear-gradient(90deg, #2d3748 0%, #1a202c 100%)")
            banner_border = "#00E676" if "BUY" in top_rec['signal'] else ("#FF5252" if "SELL" in top_rec['signal'] else "#FFD54F")

            banner_html = (
                f'<div style="background: {banner_bg}; border: 2px solid {banner_border}; border-radius: 10px; padding: 20px; margin-bottom: 22px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
                f'<div>'
                f'<span style="background-color: {banner_border}; color: #000; font-weight: bold; padding: 4px 12px; border-radius: 4px; font-size: 0.85rem; letter-spacing: 0.5px;">🏆 TOP RECOMMENDED INDEX TO TRADE TODAY</span>'
                f'<h2 style="margin: 8px 0 4px 0; color: #FFF; font-size: 1.8rem;">{top_rec["display_name"]} <span style="font-size: 1.2rem; color: #CBD5E0;">(LTP: ₹{top_rec["close"]:,.1f})</span></h2>'
                f'<div style="margin: 0; color: #E2E8F0; font-size: 0.95rem;">'
                f'Action Signal: <b>{top_rec["signal_badge"]}</b> | Profitability Score: <b style="color: #FFD54F; font-size: 1.1rem;">{top_rec["profitability_score"]}/100</b> | Conviction: <b>{top_rec["conviction"]}</b>'
                f'</div>'
                f'</div>'
                f'<div style="text-align: right; margin-top: 8px;">'
                f'<span style="font-size: 1.8rem; font-weight: bold; color: {"#00E676" if top_rec["change_pct"] >= 0 else "#FF5252"};">{top_rec["change_pct"]:+.2f}%</span><br>'
                f'<span style="color: #E2E8F0; font-size: 0.9rem;">Change: {top_rec["change_pts"]:+.1f} pts</span>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)

            # Live Technical Cards Matrix
            st.markdown("### 📊 Index Live Market & Technical Matrix")
            cols = st.columns(len(index_results))
            for i, idx in enumerate(index_results):
                with cols[i]:
                    sb = idx['signal_badge']
                    is_bull = "BUY" in sb
                    is_bear = "SELL" in sb
                    card_border = "#00E676" if is_bull else ("#FF5252" if is_bear else "#FFD54F")
                    badge_bg = "#004D25" if is_bull else ("#4D0000" if is_bear else "#4D3D00")
                    badge_color = "#00E676" if is_bull else ("#FF5252" if is_bear else "#FFD54F")

                    card_m_html = (
                        f'<div style="background-color: #1a1e24; border-radius: 8px; padding: 14px; border-top: 4px solid {card_border}; height: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<b style="color: #FFF; font-size: 1.05rem;">{idx["display_name"]}</b>'
                        f'<span style="font-size: 0.75rem; background-color: #2D3748; padding: 2px 6px; border-radius: 3px; color: #A0AEC0;">{idx["exchange"]}</span>'
                        f'</div>'
                        f'<h3 style="margin: 6px 0 2px 0; color: #FFF; font-size: 1.25rem;">₹{idx["close"]:,.1f}</h3>'
                        f'<div style="margin: 0 0 10px 0; font-size: 0.8rem; color: {"#00E676" if idx["change_pct"]>=0 else "#FF5252"};">{idx["change_pct"]:+.2f}% ({idx["change_pts"]:+.1f} pts)</div>'
                        f'<div style="background-color: {badge_bg}; color: {badge_color}; padding: 6px 8px; border-radius: 4px; font-size: 0.8rem; text-align: center; font-weight: bold; margin-bottom: 10px; border: 1px solid {card_border};">'
                        f'{idx["signal_badge"]}'
                        f'</div>'
                        f'<div style="font-size: 0.78rem; color: #CBD5E0; line-height: 1.6; background-color: #0e1117; padding: 8px; border-radius: 4px;">'
                        f'• RSI 14: <b>{idx["indicators"]["rsi_14"]}</b><br>'
                        f'• ATR 14: <b>{idx["indicators"]["atr_14"]} pts</b><br>'
                        f'• Aroon Osc: <b>{idx["indicators"]["aroon_osc"]}</b><br>'
                        f'• MACD Hist: <b>{idx["indicators"]["macd_hist"]}</b><br>'
                        f'• Lot Size: <b>{idx["lot_size"]}</b>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(card_m_html, unsafe_allow_html=True)

            st.markdown("---")

            # Two-Way Action Levels Table (Buy Above / Downside Possible Below)
            st.markdown("### 🎯 Two-Way Action Levels (Buy Above / Downside Possible Below)")
            st.caption("Quantitative conditional triggers: Exactly where to enter long momentum breakouts, where downside selloffs accelerate, and the choppy no-trade zone.")
            action_table = []
            for idx in index_results:
                al = idx.get("action_levels", {})
                atr_val = float(idx.get("indicators", {}).get("atr_14", idx["close"] * 0.015))
                buy_ab = float(al.get('buy_above', 0)) if al.get('buy_above') else round(idx['close'] * 1.002, 1)
                sell_bl = float(al.get('sell_below', 0)) if al.get('sell_below') else round(idx['close'] * 0.998, 1)
                upside_t1 = float(al.get('upside_target_1', 0)) if al.get('upside_target_1') else round(buy_ab + atr_val * 0.75, 1)
                upside_t2 = float(al.get('upside_target_2', 0)) if al.get('upside_target_2') else round(buy_ab + atr_val * 1.5, 1)
                upside_sl = float(al.get('upside_stop_loss', 0)) if al.get('upside_stop_loss') else round(buy_ab - atr_val * 0.5, 1)
                downside_t1 = float(al.get('downside_target_1', 0)) if al.get('downside_target_1') else round(sell_bl - atr_val * 0.75, 1)
                downside_t2 = float(al.get('downside_target_2', 0)) if al.get('downside_target_2') else round(sell_bl - atr_val * 1.5, 1)
                downside_sl = float(al.get('downside_stop_loss', 0)) if al.get('downside_stop_loss') else round(sell_bl + atr_val * 0.5, 1)
                chop_str = al.get('chop_zone') if al.get('chop_zone') else f"₹{sell_bl:,.1f} - ₹{buy_ab:,.1f}"

                action_table.append({
                    "Index": idx["display_name"],
                    "LTP (₹)": f"₹{idx['close']:,.1f}",
                    "🟢 BUY ABOVE": f"₹{buy_ab:,.1f}",
                    "Upside Targets": f"T1: ₹{upside_t1:,.1f} | T2: ₹{upside_t2:,.1f}",
                    "Upside SL": f"₹{upside_sl:,.1f}",
                    "Call Play (CE)": al.get('upside_option_play', f"{idx['display_name']} CE"),
                    "🔴 DOWNSIDE BELOW": f"₹{sell_bl:,.1f}",
                    "Downside Targets": f"T1: ₹{downside_t1:,.1f} | T2: ₹{downside_t2:,.1f}",
                    "Downside SL": f"₹{downside_sl:,.1f}",
                    "Put Play (PE)": al.get('downside_option_play', f"{idx['display_name']} PE"),
                    "🟡 No-Trade Chop Range": chop_str
                })
            st.dataframe(pd.DataFrame(action_table), use_container_width=True, hide_index=True)

            # Strategic Playbook Callout Cards for each Index
            st.markdown("#### 📋 Strategic Execution Playbook")
            for idx in index_results:
                al = idx.get("action_levels", {})
                sb = idx["signal_badge"]
                border_c = "#00E676" if "BUY" in sb else ("#FF5252" if "SELL" in sb else "#FFD54F")
                bg_c = "#0a2617" if "BUY" in sb else ("#2d0e12" if "SELL" in sb else "#242000")
                thesis_text = al.get('trade_thesis')
                if not thesis_text:
                    buy_ab = float(al.get('buy_above', 0)) if al.get('buy_above') else round(idx['close'] * 1.002, 1)
                    sell_bl = float(al.get('sell_below', 0)) if al.get('sell_below') else round(idx['close'] * 0.998, 1)
                    thesis_text = f"Action Levels: Buy Above ₹{buy_ab:,.1f} | Downside Possible Below ₹{sell_bl:,.1f} | Chop Zone: ₹{sell_bl:,.1f} - ₹{buy_ab:,.1f}"

                card_html = (
                    f'<div style="background-color: {bg_c}; border-left: 5px solid {border_c}; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
                    f'<b style="color: #FFF; font-size: 1.05rem;">{idx["display_name"]} — {sb}</b>'
                    f'<span style="color: #A0AEC0; font-size: 0.85rem;">Official Lot Size: <b style="color: #FFD54F;">{idx["lot_size"]}</b> | Strike Interval: <b>{idx["strike_step"]} pts</b></span>'
                    f'</div>'
                    f'<div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.5;">{thesis_text}</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            st.markdown("---")

            # Futures Trade Plans
            st.markdown("### 📈 Index Futures Trading Plans (Entry / Stop Loss / Targets)")
            f_table = []
            for idx in index_results:
                fp = idx["futures_plan"]
                f_table.append({
                    "Index": idx["display_name"],
                    "Action Signal": idx["signal_badge"],
                    "LTP (₹)": f"₹{idx['close']:,.1f}",
                    "Official Lot Size": idx["lot_size"],
                    "Day Change": f"{idx['change_pct']:+.2f}%",
                    "Entry Zone": fp["entry_zone"],
                    "Stop Loss": f"₹{fp['stop_loss']:,.1f}",
                    "Target 1": f"₹{fp['target_1']:,.1f}",
                    "Target 2": f"₹{fp['target_2']:,.1f}",
                    "R:R Ratio": fp["risk_reward_ratio"],
                    "Point Value / Lot": fp["point_value_per_lot"],
                    "Max Profit / Lot": fp["max_profit_target_1"],
                    "Conviction": idx["conviction"],
                    "Profitability Score": f"{idx['profitability_score']}/100"
                })
            st.dataframe(pd.DataFrame(f_table), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Options Explorer
            st.markdown("### 🎯 Index Options Explorer & Profitability Analysis")
            st.caption("Auto-calculated ATM, ITM, and OTM strike recommendations with estimated profit/loss per exchange lot.")
            opt_table = []
            for idx in index_results:
                for strat_item in idx["options_setup"]["strategies"]:
                    opt_table.append({
                        "Index": idx["display_name"],
                        "Strategy Play": strat_item["name"],
                        "Recommended Contract": strat_item["recommended_contract"],
                        "Target Index Points": strat_item["target_points"],
                        "Stop Loss Points": strat_item["stop_loss_points"],
                        "Est. Profit / Lot": strat_item["est_profit_lot"],
                        "Max Risk / Lot": strat_item["est_risk_lot"],
                        "Suitability": strat_item["suitability"],
                        "Rationale": strat_item["rationale"]
                    })
            st.dataframe(pd.DataFrame(opt_table), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Technical Reasons Accordion
            st.markdown("### 🔬 Technical Indicator Drivers for Each Index")
            for idx in index_results:
                with st.expander(f"📌 {idx['display_name']} - Technical Drivers & Indicator Readings"):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.write(f"**Index Profile**: {idx['description']}")
                        st.write(f"**Exchange**: {idx['exchange']} | **Strike Step**: {idx['strike_step']} pts | **Lot Size**: {idx['lot_size']}")
                        st.write(f"**ATM Strike**: `{idx['options_setup']['atm_strike']}` | **ITM**: `{idx['options_setup']['itm_strike']}` | **OTM**: `{idx['options_setup']['otm_strike']}`")
                        piv = idx.get("action_levels", {}).get("pivots", {})
                        if piv:
                            st.write(f"**Classical Pivots**: P: `₹{piv.get('pivot', 0):,.1f}` | R1: `₹{piv.get('r1', 0):,.1f}` | S1: `₹{piv.get('s1', 0):,.1f}`")
                        st.write(f"**Profitability Score**: `{idx['profitability_score']}/100`")
                    with c2:
                        st.write("**Key Technical Indicators**:")
                        ind = idx['indicators']
                        st.markdown(f"""
                        - **Moving Averages**: 20 EMA: ₹{ind['ema_20']:,.1f} | 50 EMA: ₹{ind['ema_50']:,.1f} | 200 SMA: ₹{ind['sma_200']:,.1f}
                        - **Momentum**: RSI 14: **{ind['rsi_14']}** | MACD Hist: **{ind['macd_hist']}** | Level-II Bullish: **{'YES' if ind['macd_level2'] else 'NO'}**
                        - **Aroon Indicator**: Up: **{ind['aroon_up']}** | Down: **{ind['aroon_down']}** | Oscillator: **{ind['aroon_osc']}**
                        - **Volatility**: ATR 14: **{ind['atr_14']} pts** | BB Bandwidth: **{ind['bb_bandwidth']}** | Squeeze: **{'YES' if ind['is_squeezing'] else 'NO'}**
                        """)
                        st.write("**Algorithmic Reasons**:")
                        for r in idx["technical_reasons"]:
                            st.write(f"• {r}")

    with tab_ai:
        st.subheader("Actionable Trade Plans & Risk Management")
        if not all_candidates:
            st.info("No candidates met the strict screening criteria today. Markets may be consolidating.")
        else:
            advisor = AIAdvisor(api_key=gemini_key_input if gemini_key_input else None)
            engine_name = "Google Gemini Flash (Live LLM)" if advisor.is_ai_ready else "Algorithmic Precision Engine"
            st.caption(f"Trade plans generated via **{engine_name}**")

            with st.spinner("Generating risk-reward evaluations and trade thesis..."):
                plans = advisor.analyze_candidates(all_candidates)

            # Display cards in 2 columns
            cols = st.columns(2)
            for idx, p in enumerate(plans):
                col = cols[idx % 2]
                with col:
                    action = p.get('trade_action', 'BUY').upper()
                    is_bullish = ("BUY" in action or action == "LONG") and "PE" not in action
                    badge_color = "#00E676" if is_bullish else "#FF5252"
                    badge_text_color = "#000" if is_bullish else "#FFF"
                    border_color = badge_color

                    # Column labels based on trade direction (Shorts profit downward)
                    t1_lbl = "Target 1 (Book 33%)" if is_bullish else "Downside T1 (Cover 33%)"
                    t2_lbl = "Target 2 (Book 33%)" if is_bullish else "Downside T2 (Cover 33%)"
                    t3_lbl = "Target 3 (Trail SL)" if is_bullish else "Downside T3 (Trail SL)"
                    t4_lbl = "Target 4 (Extended)" if is_bullish else "Downside T4 (Extended)"
                    sl_lbl = "Stop Loss (Base)" if is_bullish else "Stop Loss (Invalidation)"

                    # Commodity contract unit badge
                    sym = p.get("symbol", "")
                    unit_badge = ""
                    if UniverseManager.is_commodity(sym):
                        comm_map = {
                            "GOLD": "MCX Futures (₹ / 10g)",
                            "SILVER": "MCX Futures (₹ / kg)",
                            "CRUDEOIL": "MCX Futures (₹ / bbl)",
                            "NATURALGAS": "MCX Futures (₹ / mmBtu)",
                            "COPPER": "MCX Futures (₹ / kg)",
                            "ZINC": "MCX Futures (₹ / kg)",
                            "ALUMINIUM": "MCX Futures (₹ / kg)",
                        }
                        unit_badge = f' | Contract: <b style="color: #FFD54F;">{comm_map.get(sym, "MCX Commodity")}</b>'

                    # Short / Put option explainer banner
                    short_banner = ""
                    if not is_bullish:
                        short_banner = (
                            f'<div style="background-color: #2b0d10; border-left: 4px solid #FF5252; padding: 6px 12px; border-radius: 4px; font-size: 0.8rem; color: #FFCDD2; margin-bottom: 8px;">'
                            f'<b>📉 Short / Bearish Setup:</b> Profit is booked as price declines below Entry (₹{p.get("entry_price")}) toward Downside Targets.'
                            f'</div>'
                        )

                    trailing_badge = (
                        f'<div style="background-color: #242000; border-left: 4px solid #FFD54F; padding: 8px 12px; border-radius: 4px; font-size: 0.82rem; color: #FFF8E1; margin-bottom: 8px;">'
                        f'<b>💡 Trailing Playbook:</b> {p.get("trailing_playbook")}</div>'
                    ) if p.get("trailing_playbook") else ''

                    thesis_clean = p.get('thesis', '').replace('"', '&quot;')

                    card_html = (
                        f'<div style="background-color: #1a1e24; border-radius: 10px; padding: 18px; margin-bottom: 16px; border-left: 6px solid {border_color}; box-shadow: 0 4px 8px rgba(0,0,0,0.35);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<h3 style="margin: 0; color: #FFF; font-size: 1.3rem;">{p.get("symbol")}</h3>'
                        f'<span style="background-color: {badge_color}; color: {badge_text_color}; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85rem;">{action}</span>'
                        f'</div>'
                        f'<p style="margin: 6px 0 10px 0; color: #8892B0; font-size: 0.85rem;">'
                        f'Category: <b style="color: #64B5F6;">{p.get("category")}</b>{unit_badge} | '
                        f'Timeframe: <b style="color: #FFF;">{p.get("timeframe")}</b> | '
                        f'Conviction: <b style="color: #FFD54F;">{p.get("conviction")}</b>'
                        f'</p>'
                        f'{short_banner}'
                        f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; background-color: #0e1117; padding: 12px; border-radius: 6px; margin: 10px 0 6px 0;">'
                        f'<div><span style="color: #888; font-size: 0.75rem;">Entry Trigger</span><br><b style="color: #FFF; font-size: 1.05rem;">₹{p.get("entry_price")}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">{sl_lbl}</span><br><b style="color: #FF5252; font-size: 1.05rem;">₹{p.get("tight_stop_loss", p.get("stop_loss"))}</b><br><span style="font-size: 0.7rem; color: #A0AEC0;">Swing Floor: ₹{p.get("conservative_stop_loss", p.get("stop_loss"))}</span></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">{t1_lbl}</span><br><b style="color: #00E676; font-size: 1.05rem;">₹{p.get("target_1")}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">{t2_lbl}</span><br><b style="color: #00E676; font-size: 1.05rem;">₹{p.get("target_2")}</b></div>'
                        f'</div>'
                        f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; background-color: #12161f; padding: 8px 12px; border-radius: 6px; margin-bottom: 8px;">'
                        f'<div><span style="color: #888; font-size: 0.75rem;">{t3_lbl}</span><br><b style="color: #64B5F6; font-size: 0.95rem;">₹{p.get("target_3", p.get("target_2"))}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">{t4_lbl}</span><br><b style="color: #BA68C8; font-size: 0.95rem;">₹{p.get("target_4", p.get("target_2"))}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">R:R Ratio</span><br><b style="color: #FFD54F; font-size: 0.85rem;">{p.get("risk_reward_ratio")}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.75rem;">Position Sizing</span><br><b style="color: #FFF; font-size: 0.85rem;">{p.get("shares_for_2k_risk", "—")} sh (₹2k risk)</b></div>'
                        f'</div>'
                        f'{trailing_badge}'
                        f'<div style="margin: 8px 0 0 0; color: #E0E0E0; font-size: 0.88rem; line-height: 1.45;"><i>&ldquo;{thesis_clean}&rdquo;</i></div>'
                        f'</div>'
                    )
                    render_html(card_html)

                    # Quick 1-click Add to Item Tracker
                    current_user_name = get_current_user() or "trader"
                    with st.expander(f"📌 Track {p.get('symbol')} in My Item Tracker"):
                        with st.form(f"track_form_{idx}_{p.get('symbol')}", clear_on_submit=False):
                            tr_c1, tr_c2 = st.columns(2)
                            tr_buy = tr_c1.number_input("Entry Price (₹)", value=float(p.get("entry_price", 100.0)), key=f"tr_buy_{idx}")
                            default_qty = 50.0
                            sz = str(p.get("shares_for_2k_risk", ""))
                            if sz.isdigit():
                                default_qty = float(sz)
                            tr_qty = tr_c2.number_input("Quantity / Units", value=default_qty, min_value=1.0, key=f"tr_qty_{idx}")
                            
                            tr_c3, tr_c4 = st.columns(2)
                            raw_cat = str(p.get("category", "")).upper()
                            def_type_idx = 0
                            if "OPTION" in raw_cat or "PE" in action or "CE" in action:
                                def_type_idx = 2 if "PE" in action else 1
                            elif UniverseManager.is_commodity(p.get("symbol", "")):
                                def_type_idx = 4
                            tr_type = tr_c3.selectbox("Asset Type", ["EQUITY", "OPTION_CE", "OPTION_PE", "FUTURES", "COMMODITY", "INDEX"], index=def_type_idx, key=f"tr_type_{idx}")
                            tr_sl = tr_c4.number_input("Stop Loss (₹)", value=float(p.get("tight_stop_loss", p.get("stop_loss", tr_buy * 0.95))), key=f"tr_sl_{idx}")
                            
                            tr_tgt = float(p.get("target_1", tr_buy * 1.10))
                            tr_notes = f"{p.get('category')} setup - {p.get('thesis', '')[:80]}"
                            
                            tr_submit = st.form_submit_button(f"➕ Track {p.get('symbol')} in Live Portfolio", type="primary", use_container_width=True)
                            if tr_submit:
                                PortfolioTracker.add_position(
                                    username=current_user_name,
                                    symbol=p.get("symbol"),
                                    buy_price=float(tr_buy),
                                    qty=float(tr_qty),
                                    asset_type=tr_type,
                                    stop_loss=float(tr_sl),
                                    target=float(tr_tgt),
                                    notes=tr_notes
                                )
                                st.toast(f"✅ Added {p.get('symbol')} to {current_user_name}'s Item Tracker!", icon="📌")
                                time.sleep(0.4)
                                st.rerun()

    with tab_screeners:
        for cat, c_list in results.items():
            st.subheader(f"📌 {cat} Setups ({len(c_list)})")
            if c_list:
                table_data = []
                for c in c_list:
                    vol_2x_badge = "🔥 YES (>2X)" if c.get("vol_confirmed_2x") else "—"
                    rvol_9_str = f"{c.get('rvol_9', c['rvol']):.1f}x"
                    
                    retest_val = c.get("retest_support_50pct", 0)
                    retest_str = f"₹{retest_val:,.1f}" if retest_val > 0 else "—"
                    
                    sector_val = c.get("sector", "Broad")
                    tailwind_badge = f"{sector_val} 🌊" if c.get("has_sector_tailwind") else sector_val
                    panic_badge = "🛡️ YES" if c.get("panic_day_resilient") else "—"

                    comm_unit = ""
                    sym_c = c["symbol"]
                    if UniverseManager.is_commodity(sym_c):
                        comm_unit_map = {
                            "GOLD": " (₹/10g)",
                            "SILVER": " (₹/kg)",
                            "CRUDEOIL": " (₹/bbl)",
                            "NATURALGAS": " (₹/mmBtu)",
                            "COPPER": " (₹/kg)",
                            "ZINC": " (₹/kg)",
                            "ALUMINIUM": " (₹/kg)",
                        }
                        comm_unit = comm_unit_map.get(sym_c, " (MCX)")

                    row = {
                        "Symbol": c["symbol"],
                        "Score": c["score"],
                        "LTP (₹)": f"₹{c['close']:,.2f}{comm_unit}" if comm_unit else c["close"],
                        "Change %": f"{c['change_pct']:+.2f}%",
                        "50% Retest Level": retest_str,
                        "Sector Tailwind": tailwind_badge,
                        "Panic Resilient": panic_badge,
                        "Vol Confirmed (>2X)": vol_2x_badge,
                        "RVol (9d)": rvol_9_str,
                        "RVol (20d)": f"{c['rvol']:.1f}x",
                        "RSI (14)": c["rsi"],
                        "Key Triggers": ", ".join(c.get("reasons", []))
                    }
                    if "recommended_strike" in c:
                        row["Rec. Strike"] = f"{c['recommended_strike']:.0f} {c.get('option_type', '')}"
                    table_data.append(row)
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.caption("No stocks met this specific category's thresholds.")

    with tab_tracker:
        render_item_tracker_view(current_user=get_current_user() or "trader", data_dict=data_dict)

    with tab_charts:
        st.subheader("Interactive Candlestick & Technical Inspector")
        if all_candidates:
            symbols_to_chart = list(dict.fromkeys([c["symbol"] for c in all_candidates]))
            selected_sym = st.selectbox("Select Candidate to Inspect", symbols_to_chart)
            if selected_sym in data_dict:
                df_chart = enrich_with_indicators(data_dict[selected_sym].tail(120).copy())

                comm_chart_tag = ""
                if UniverseManager.is_commodity(selected_sym):
                    comm_chart_map = {
                        "GOLD": " (MCX ₹ / 10g)",
                        "SILVER": " (MCX ₹ / kg)",
                        "CRUDEOIL": " (MCX ₹ / bbl)",
                        "NATURALGAS": " (MCX ₹ / mmBtu)",
                        "COPPER": " (MCX ₹ / kg)",
                        "ZINC": " (MCX ₹ / kg)",
                        "ALUMINIUM": " (MCX ₹ / kg)",
                    }
                    comm_chart_tag = comm_chart_map.get(selected_sym, " (MCX)")

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    subplot_titles=(f"{selected_sym} — Price & Moving Averages{comm_chart_tag}", "Volume & RVol"),
                    row_heights=[0.75, 0.25]
                )

                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df_chart.index,
                    open=df_chart['open'],
                    high=df_chart['high'],
                    low=df_chart['low'],
                    close=df_chart['close'],
                    name='Candles'
                ), row=1, col=1)

                # EMAs
                if 'ema_20' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ema_20'], line=dict(color='#FFD54F', width=1.5), name='EMA 20'), row=1, col=1)
                if 'ema_50' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['ema_50'], line=dict(color='#29B6F6', width=1.5), name='EMA 50'), row=1, col=1)
                if 'bb_upper' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['bb_upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['bb_lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

                # Volume Bars
                vol_colors = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df_chart['close'], df_chart['open'])]
                fig.add_trace(go.Bar(
                    x=df_chart.index,
                    y=df_chart['volume'],
                    marker_color=vol_colors,
                    name='Volume'
                ), row=2, col=1)

                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=600,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run a screen to view candidate technical charts.")

    with tab_docs:
        st.markdown(r"""
        ### 📐 Strategy Logics & Quantitative Rules
        
        **🔥 1. Institutional Volume Confirmation Architecture**:
        Across all 6 quantitative strategies, institutional volume confirmation is triggered when:
        $$\text{Current Volume} \ge 2.0 \times \text{9-Day Volume SMA} \quad (\text{RVol}_{9d} \ge 2.0\text{X})$$
        - **Breakout & BTST**: Confirms genuine institutional accumulation and expansion out of tight multi-week consolidation.
        - **Breakdown**: Confirms aggressive institutional distribution / panic dumping below key support levels.
        - **Swing Reversal**: Confirms strong buyer absorption and aggressive demand defense at dynamic EMA 20/50 support.
        - **Options (CE/PE)**: Ensures volume burst to drive delta expansion and overcome theta decay.

        **🛡️ 2. Retracement Check: 50% Breakout Candle Body Defense Rule**:
        $$\text{Candle Body} = |\text{Close} - \text{Open}| \implies \text{Defense Support Level} = \text{Open} + 0.50 \times (\text{Close} - \text{Open})$$
        - **Institutional Principle**: Smart money actively protects the upper 50% of an expansion bar.
        - **Healthy Retracement**: When price pulls back to retest the breakout zone, intraday wicks or Day-2 dips must hold **above** the 50% candle body midpoint.
        - **Optimal Dip-Buy Zone**: The range between the current Close and the 50% body level ($\text{Support}_{50\%} \le \text{Entry} \le \text{Close}$) offers the highest risk-reward entry with an invalidation stop-loss just below $\text{Support}_{50\%}$.
        - **Invalidation Trigger**: If sellers penetrate deeper than 50% into the candle body, the breakout is marked as compromised (high bull-trap risk).

        **🌊 3. Sector Tailwind & Panic-Day Outperformance**:
        - **Sector Tailwind**: Maps each stock to its respective benchmark (e.g., TCS/INFY $\to$ **NIFTY IT**, HDFCBANK/SBIN $\to$ **BANK NIFTY**, Midcaps $\to$ **MIDCPNIFTY**, Equities $\to$ **NIFTY 50**). Confirms whether the parent sector is trending bullish ($\text{Close} \ge \text{EMA}_{20}$, $\text{RSI} \ge 50$) to ensure high probability momentum alignment.
        - **Panic-Day Resilience (Smart Money Footprint)**: Identifies broad market "Panic Days" (sessions where NIFTY drops $\le -0.75\%$).
          - Stocks that closed **GREEN** while NIFTY plummeted indicate massive institutional demand soaking up all retail panic supply.
          - High Alpha stocks ($\text{Stock Return} - \text{NIFTY Return} \ge +1.0\%$) are tagged with `🛡️ Panic-Day Resilient` (+15 pts score bonus).
        - **Mansfield Relative Strength (MRS)**: Weinstein's 50-period relative performance vs NIFTY 50:
          $$\text{RS} = \frac{\text{Stock Close}}{\text{NIFTY Close}}, \quad \text{MRS} = \left(\frac{\text{RS}}{\text{SMA}(\text{RS}, 50)} - 1.0\right) \times 100$$
          Positive MRS indicates true market leaders in Stage-2 accumulation.

        ### 🎯 Strategy Frameworks:
        1. **Multi-Week & 52W Breakout**:
           - Proximity to 52-Week High ($\le 4\%$) or clearing 20-day swing resistance.
           - Stage-2 Uptrend: $\text{Price} \ge \text{EMA}_{20} \ge \text{EMA}_{50}$.
           - Bollinger Bandwidth volatility contraction / squeeze expansion.
           - Volume Confirmation: $\ge 2.0\times$ 9-day average volume (+15 pts score bonus).
           - 50% Breakout Candle Body Defense support calculation.

        2. **Multi-Week & Support Breakdown**:
           - Proximity to 52-Week Low or violation of 20-day swing support.
           - Stage-4 Downtrend: $\text{Price} \le \text{EMA}_{20} \le \text{EMA}_{50}$.
           - Weak Close Location Value ($\text{CLV} \le 0.35$), sellers driving through the close.
           - Bearish RSI breakdown ($\le 45$) and Institutional Volume Confirmation ($\ge 2.0\times$ 9d avg).
           - 50% Breakdown Candle Body Ceiling resistance calculation.

        3. **Swing Pullback & Value**:
           - Pullback towards high-value dynamic support (EMA 20 / EMA 50 / lower Bollinger Band).
           - Bullish reversal candlestick: Hammer, long lower absorption wick, or $\text{CLV} \ge 0.55$.
           - RSI reset to healthy continuation zone ($38 \le \text{RSI} \le 64$).
           - Institutional buyer absorption volume ($\ge 2.0\times$ 9-day average volume).
           - Target previous swing high with minimum $1:2.5$ Risk:Reward ratio.

        4. **BTST (Buy Today, Sell Tomorrow)**:
           - Accumulation into the close: Close Location Value $\text{CLV} \ge 0.70$ (top 30% of day's range).
           - Volume Confirmation: $\ge 2.0\times$ 9-day average volume (+20 pts bonus).
           - Testing or breaking fresh 5-day swing high.

        5. **Intraday Momentum**:
           - Directional expansion: Range reaching $\ge 50\%$ of 14-day daily ATR.
           - Trend alignment with EMA 9 and EMA 20.
           - Strong directional CLV ($\ge 0.68$ for Long, $\le 0.32$ for Short) and Volume Confirmation ($\ge 2.0\times$ 9d avg).

        6. **Options (CE & PE Buildup)**:
           - Call (CE) buying: Bullish thrust, elevated volume, $\text{RSI} \ge 54$, $\text{CLV} \ge 0.65$.
           - Put (PE) buying: Bearish drop, elevated volume, $\text{RSI} \le 46$, $\text{CLV} \le 0.35$.
           - Options Volume Confirmation ($\ge 2.0\times$ 9d avg) for delta thrust.
           - Dynamic ATM / ITM / OTM strike calculations with Indian stock step intervals.
        """)

    with tab_audit:
        st.subheader("🔍 API Raw Data & Indicator Proof (Admin Audit)")
        st.caption("Inspect exact timestamps, OHLCV candles, and computed mathematical formulas to cross-verify against NSE/MCX.")
        
        # External API Health Diagnostic Panel
        st.markdown("### 🌐 External Data Feed & API Connectivity Health")
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        h_col1.metric("API Call Status", "🟢 Connected" if not data_health.get("api_call_failed") else "⚠️ Failed")
        h_col2.metric("Data Freshness", "LATEST (LIVE)" if is_live else "NOT LATEST (CACHED)")
        h_col3.metric("Loaded Symbols", f"{data_health.get('total_symbols_loaded', len(data_dict))}/{data_health.get('total_symbols_requested', len(symbols))}")
        h_col4.metric("Fallback / Failed", f"{len(data_health.get('fallback_symbols', []))} / {len(data_health.get('failed_symbols', []))}")

        if data_health.get("failure_reasons"):
            st.error("⚠️ **External API Failure Diagnostic Logs:**")
            for err in data_health.get("failure_reasons", []):
                st.code(err, language="text")

        st.markdown("---")

        # Download button for API_DATA_AUDIT_REPORT.txt
        audit_file_path = Path(__file__).resolve().parent.parent / "API_DATA_AUDIT_REPORT.txt"
        if audit_file_path.exists():
            with open(audit_file_path, "r", encoding="utf-8") as f:
                audit_text = f.read()
            st.download_button(
                label="📥 Download Full API Data Audit Report (.txt)",
                data=audit_text,
                file_name="API_DATA_AUDIT_REPORT.txt",
                mime="text/plain",
                type="primary"
            )

        # Interactive asset inspector
        if all_candidates:
            inspect_sym = st.selectbox("Select Asset to Audit", [c['symbol'] for c in all_candidates], key="audit_sym_picker")
            if inspect_sym in data_dict:
                raw_df = data_dict[inspect_sym]
                enriched_df = enrich_with_indicators(raw_df.copy())
                last_candle = enriched_df.iloc[-1]
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Close Price", f"₹{last_candle['close']:.2f}")
                c2.metric("Relative Volume", f"{last_candle.get('rvol_20', 1.0):.2f}x")
                c3.metric("RSI (14)", f"{last_candle.get('rsi_14', 50):.1f}")
                c4.metric("CLV (Close Location)", f"{last_candle.get('clv', 0.5):.2f}")
                
                st.write("**Recent Raw Candlesticks from API (Last 10 sessions):**")
                st.dataframe(raw_df.tail(10), use_container_width=True)
                
                st.write("**Computed Vectorized Indicators (Last 10 sessions):**")
                ind_cols = ['close', 'ema_20', 'ema_50', 'sma_200', 'rsi_14', 'bb_bandwidth', 'atr_14', 'rvol_20', 'clv']
                avail_cols = [c for c in ind_cols if c in enriched_df.columns]
                st.dataframe(enriched_df[avail_cols].tail(10), use_container_width=True)
        else:
            st.info("Run a screen to inspect candidate raw data and indicators.")

else:
    st.info("👈 Select your options on the sidebar and click **'Run Live Market Screen'** to begin market scans, or review and manage your live tracked positions below:")
    render_item_tracker_view(current_user=get_current_user() or "trader", data_dict={})
