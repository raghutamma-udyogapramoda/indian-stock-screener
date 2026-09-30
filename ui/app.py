"""
Streamlit Web Dashboard for Indian Equities Screener & AI Trade Advisor.
Supports: BTST, Breakout, Breakdown, Swing, Intraday, and Options (CE/PE).
Run with: streamlit run ui/app.py
"""

import time
from datetime import datetime, timezone, timedelta
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root and current working directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

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

# Robust import for core.auth with filesystem fallback
try:
    from core.auth import render_login_gate, render_sidebar_user_badge, get_current_user, is_admin
except Exception:
    import importlib.util
    _auth_file = PROJECT_ROOT / "core" / "auth.py"
    _spec = importlib.util.spec_from_file_location("core.auth", str(_auth_file))
    _auth_mod = importlib.util.module_from_spec(_spec)
    sys.modules["core.auth"] = _auth_mod
    _spec.loader.exec_module(_auth_mod)
    render_login_gate = _auth_mod.render_login_gate
    render_sidebar_user_badge = _auth_mod.render_sidebar_user_badge
    get_current_user = _auth_mod.get_current_user
    is_admin = _auth_mod.is_admin

# Guard against Streamlit Cloud stale in-memory module caching across git updates
import importlib
_APP_BUILD_SIG = "2026_09_30_realtime_v6_twoway_bias_fix"
if sys.modules.get("__CURRENT_BUILD_SIG__") != _APP_BUILD_SIG:
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == pkg or mod_name.startswith(pkg + ".") for pkg in ("core", "screeners", "ai", "providers")):
            sys.modules.pop(mod_name, None)
    sys.modules["__CURRENT_BUILD_SIG__"] = _APP_BUILD_SIG

# Ensure all core and screener modules are strictly fresh
try:
    import core.universe
    import core.macro
    import core.relative_strength
    import core.stock_levels
    import core.indicators
    import core.tracker
    import screeners.dip_leaders
    import screeners.breakout
    import screeners.breakdown
    import screeners.swing
    import screeners.btst
    import screeners.intraday
    import screeners.options_fno
    importlib.reload(core.universe)
    importlib.reload(core.macro)
    importlib.reload(core.relative_strength)
    importlib.reload(core.stock_levels)
    importlib.reload(core.indicators)
    importlib.reload(core.tracker)
    importlib.reload(screeners.dip_leaders)
    importlib.reload(screeners.breakout)
    importlib.reload(screeners.breakdown)
    importlib.reload(screeners.swing)
    importlib.reload(screeners.btst)
    importlib.reload(screeners.intraday)
    importlib.reload(screeners.options_fno)
except Exception:
    pass

from ai.advisor import AIAdvisor
from core.indices import IndexDerivativesAnalyzer, INDEX_SPECS
from core.macro import MacroMarketEngine
from core.tracker import PortfolioTracker
from core.universe import UniverseManager
from core.indicators import enrich_with_indicators
from core.stock_levels import StockLevelAnalyzer
from providers.breeze_provider import BreezeProvider
from providers.breeze_static_provider import BreezeStaticProvider
from providers.nse_direct_provider import NSEDirectProvider
from providers.yfinance_provider import YahooFinanceProvider
from screeners.breakdown import BreakdownScreener
from screeners.breakout import BreakoutScreener
from screeners.btst import BTSTScreener
from screeners.dip_leaders import DipLeaderScreener
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


@st.fragment(run_every="10s")
def render_live_tracked_positions_fragment(target_user: str, data_dict: Dict[str, pd.DataFrame]):
    """
    Reruns automatically every 10 seconds to check real-time stock prices
    and dynamically update recommendations (BUY, HOLD, EXIT, AVERAGE DOWN).
    """
    positions = PortfolioTracker.load_positions(target_user)
    
    # Live Stream Status Strip
    c_status, c_time, c_refresh = st.columns([2.5, 2.2, 1.3])
    with c_status:
        st.markdown(
            '<div style="display: flex; align-items: center; gap: 8px; margin-top: 4px;">'
            '<span style="height: 10px; width: 10px; background-color: #00E676; border-radius: 50%; display: inline-block; box-shadow: 0 0 10px #00E676;"></span>'
            '<b style="color: #00E676; font-size: 0.88rem;">LIVE 10s STREAM ACTIVE</b>'
            '<span style="color: #787b86; font-size: 0.8rem;">(auto-checking prices & recos)</span>'
            '</div>',
            unsafe_allow_html=True
        )
    with c_time:
        now_time = time.strftime("%I:%M:%S %p")
        st.markdown(f"<div style='text-align: right; color: #A0AEC0; font-size: 0.82rem; margin-top: 4px;'>⏱️ Last Checked: <b style='color: #FFFFFF;'>{now_time}</b></div>", unsafe_allow_html=True)
    with c_refresh:
        if st.button("🔄 Sync Now", use_container_width=True, type="secondary", key=f"btn_sync_now_{target_user}", help="Instantly pull latest exchange price"):
            st.rerun(scope="fragment")

    if not positions:
        st.info("💡 You have no tracked positions yet. Use the form above to add stocks/options you bought, or click **'📌 Track this Position'** on any setup in the **AI Trade Plans** tab.")
        return

    # 1. Fetch instantaneous quotes for all tracked positions
    needed_syms = []
    for p in positions:
        und = p.get("underlying") or PortfolioTracker.get_underlying_symbol(p.get("symbol", ""))
        if und:
            needed_syms.append(und)
            needed_syms.append(p.get("symbol", ""))
    
    needed_syms = list(dict.fromkeys(needed_syms))
    live_quotes = PortfolioTracker.fetch_live_quotes(needed_syms)

    # 2. Fetch historical candle data for any symbols not in data_dict
    active_data = dict(data_dict)
    missing_history = [s for s in needed_syms if s not in active_data]
    if missing_history:
        try:
            yf_fallback = YahooFinanceProvider(cache_ttl_hours=1.0)
            fetched = yf_fallback.fetch_batch_ohlcv(missing_history, period="6mo", interval="1d", max_workers=5)
            for s, df in fetched.items():
                if df is not None and not df.empty:
                    active_data[s] = df
        except Exception:
            pass

    # 3. Dynamic Signal & Recommendation Shift Detection
    reco_cache_key = f"prev_recos_{target_user}"
    if reco_cache_key not in st.session_state:
        st.session_state[reco_cache_key] = {}
    prev_recos = st.session_state[reco_cache_key]

    reco_shifts = []
    evaluated = []

    for p in positions:
        und = p.get("underlying") or PortfolioTracker.get_underlying_symbol(p.get("symbol", ""))
        df_for_pos = active_data.get(und)
        if df_for_pos is None or (isinstance(df_for_pos, pd.DataFrame) and df_for_pos.empty):
            df_for_pos = active_data.get(p.get("symbol", ""))

        quote = live_quotes.get(und) or live_quotes.get(p.get("symbol", ""))
        res = PortfolioTracker.evaluate_live_position(p, live_df=df_for_pos, live_quote=quote)
        evaluated.append(res)

        pos_id = p["id"]
        cur_action = res["action_badge"]
        if pos_id in prev_recos and prev_recos[pos_id] != cur_action:
            reco_shifts.append({
                "symbol": p["symbol"],
                "old": prev_recos[pos_id],
                "new": cur_action,
                "price": res["current_price"],
                "color": res["action_color"]
            })
        prev_recos[pos_id] = cur_action

    # If any recommendation changed dynamically between ticks, show real-time alert banner!
    if reco_shifts:
        for shift in reco_shifts:
            st.markdown(
                f'<div style="background: rgba(255, 152, 0, 0.15); border-left: 5px solid {shift["color"]}; border-radius: 6px; padding: 10px 14px; margin: 10px 0; color: #FFFFFF; font-size: 0.9rem;">'
                f'⚡ <b>Real-Time Signal Shift:</b> <b>{shift["symbol"]}</b> recommendation changed from <code>{shift["old"]}</code> ➔ <b style="color: {shift["color"]};">{shift["new"]}</b> at CMP ₹{shift["price"]:,.2f}!'
                f'</div>',
                unsafe_allow_html=True
            )

    # 4. Portfolio Summary KPIs
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
    kpi4.metric("Active Positions", f"{len(positions)} Tracked", delta=f"{len(evaluated)} Live")

    # Action distribution summary
    if evaluated:
        n_avg = sum(1 for r in evaluated if r["action_type"] == "AVERAGE_DOWN")
        n_hold = sum(1 for r in evaluated if "HOLD" in r["action_type"])
        n_fake_rise = sum(1 for r in evaluated if r["action_type"] == "FAKE_RISE")
        n_sl = sum(1 for r in evaluated if r["action_type"] == "EXIT")
        n_tgt = sum(1 for r in evaluated if r["action_type"] == "TARGET_HIT")

        st.markdown(
            f'<div style="background: #131722; border: 1px solid #2a2e39; border-radius: 8px; padding: 10px 16px; margin: 12px 0 20px 0; font-size: 0.85rem;">'
            f'<b>Live Decision Distribution:</b> '
            f'<span style="color: #00E676; margin-right: 14px;">🟢 Average Down (Fake Fall): <b>{n_avg}</b></span> '
            f'<span style="color: #64B5F6; margin-right: 14px;">🟢 Strong Hold: <b>{n_hold}</b></span> '
            f'<span style="color: #FF9800; margin-right: 14px;">⚠️ Fake Rise Alert: <b>{n_fake_rise}</b></span> '
            f'<span style="color: #FF5252; margin-right: 14px;">🛑 Exit (SL Broken): <b>{n_sl}</b></span> '
            f'<span style="color: #FFD54F;">🎯 Target Hit: <b>{n_tgt}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # 5. Position Cards
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

        day_chg = item.get("day_change", 0.0)
        day_chg_pct = item.get("day_change_pct", 0.0)
        chg_arrow = "▲" if day_chg >= 0 else "▼"
        chg_color = "#00E676" if day_chg >= 0 else "#FF5252"

        # Callout card
        comm_unit_badge = ""
        unit_lbl = ""
        if item.get("commodity_unit"):
            comm_unit_badge = f'<span style="background-color: #E65100; color: #FFF; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">{item["commodity_unit"]}</span>'
            unit_lbl = f" ({item['commodity_unit']})"

        card_html = (
            f'<div style="background-color: {bg_color}; border-left: 6px solid {border_color}; border-radius: 10px; padding: 18px; margin-bottom: 12px; border-top: 1px solid #2a2e39; border-right: 1px solid #2a2e39; border-bottom: 1px solid #2a2e39; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">'
            f'<div>'
            f'<span style="background-color: #2962FF; color: #FFF; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px;">{item["asset_type"]}</span>'
            f'{comm_unit_badge}'
            f'<span style="font-size: 1.35rem; font-weight: bold; color: #FFF;">{item["symbol"]}</span> '
            f'<span style="color: #787b86; font-size: 0.85rem; margin-left: 8px;">Bought: {item["buy_date"]}</span>'
            f'</div>'
            f'<div style="text-align: right;">'
            f'<span style="background-color: {badge_color}; color: #000; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.85rem;">{item["action_badge"]}</span>'
            f'</div>'
            f'</div>'
            f'<div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; background-color: #0e1117; padding: 12px; border-radius: 6px; margin: 14px 0 10px 0;">'
            f'<div><span style="color: #787b86; font-size: 0.75rem;">Buy Price</span><br><b style="color: #FFF; font-size: 1.05rem;">₹{item["buy_price"]:,.2f}</b><span style="color: #A0AEC0; font-size: 0.7rem;">{unit_lbl}</span></div>'
            f'<div><span style="color: #787b86; font-size: 0.75rem;">Live CMP (10s)</span><br><b style="color: #00E676; font-size: 1.05rem;">₹{item["current_price"]:,.2f}</b><span style="color: #A0AEC0; font-size: 0.7rem;">{unit_lbl}</span> <span style="color: {chg_color}; font-size: 0.78rem;">{chg_arrow} {day_chg_pct:+.2f}%</span></div>'
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
        st.markdown(card_html, unsafe_allow_html=True)

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
                    PortfolioTracker.delete_position(target_user, pos_id)
                    st.toast(f"Removed {item['symbol']} from {target_user.upper()}'s tracker.", icon="🗑️")
                    time.sleep(0.3)
                    st.rerun()


def render_item_tracker_view(current_user: str, data_dict: Dict[str, pd.DataFrame]):
    """Renders the comprehensive Live Item Tracker for the authenticated user."""
    target_user = current_user
    if is_admin(current_user):
        admin_user_options = ["guruteja", "raghavendra", "naresh", "admin", "trader"]
        if current_user.lower() not in admin_user_options:
            admin_user_options.insert(0, current_user.lower())
        curr_idx = admin_user_options.index(current_user.lower()) if current_user.lower() in admin_user_options else 0
        
        adm_box_c1, adm_box_c2 = st.columns([2.5, 1.5])
        with adm_box_c1:
            st.markdown(
                '<div style="background: rgba(255, 215, 0, 0.08); border-left: 4px solid #FFD700; border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.85rem; color: #FFF8E1;">'
                '<b>👑 Administrator Portfolio Management:</b> You have full admin access. Select any user below to inspect or manage their live portfolio.'
                '</div>',
                unsafe_allow_html=True
            )
        with adm_box_c2:
            target_user = st.selectbox(
                "Select Portfolio to View/Manage",
                admin_user_options,
                index=curr_idx,
                format_func=lambda u: f"👑 {u.upper()} (Admin)" if is_admin(u) else f"👤 {u.upper()} (Trader)",
                key=f"admin_portfolio_selector_{current_user}"
            )

    positions = PortfolioTracker.load_positions(target_user)

    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        admin_tag = " <span style='color: #FFD700; font-size: 0.85rem; border: 1px solid #FFD700; padding: 2px 8px; border-radius: 12px;'>👑 ADMIN</span>" if is_admin(target_user) else ""
        st.markdown(f"<h3 style='margin-bottom: 2px;'>📌 {target_user.upper()}'s Live Item Tracker & Portfolio Watch{admin_tag}</h3>", unsafe_allow_html=True)
        st.caption("Real-time position monitoring with intelligent BUY, HOLD, EXIT, and AVERAGE DOWN (Fake Fall / Shakeout Detection) signals. Automatically checks live prices every 10 seconds.")
    with h_col2:
        if positions:
            if st.button("🧹 Clear All Positions", type="secondary", use_container_width=True, key=f"clear_all_{target_user}", help="Remove all tracked items"):
                PortfolioTracker.clear_all_positions(target_user)
                st.toast(f"✅ Cleared all tracked positions for {target_user.upper()}.", icon="🧹")
                time.sleep(0.3)
                st.rerun()

    # Form: Add Position Manually (Kept outside fragment so form typing is never interrupted by 10s auto-refresh)
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
                        username=target_user,
                        symbol=new_sym.strip(),
                        buy_price=float(new_buy),
                        qty=float(new_qty),
                        asset_type=new_type,
                        stop_loss=float(new_sl) if new_sl > 0 else None,
                        target=float(new_tgt) if new_tgt > 0 else None,
                        notes=new_notes
                    )
                    st.toast(f"✅ Added {new_sym.upper()} to {target_user.upper()}'s tracker!", icon="📌")
                    time.sleep(0.4)
                    st.rerun()

    st.markdown("---")

    # Render the 10-second auto-refreshing live positions fragment
    render_live_tracked_positions_fragment(target_user=target_user, data_dict=data_dict)


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
        "First-to-Recover Dip Leaders",
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

cache_ttl_option = st.sidebar.select_slider(
    "Data Freshness (Cache TTL)",
    options=["Live (0m)", "15m", "30m", "1h", "2h", "4h"],
    value="Live (0m)",
    help="Determines how long market data remains valid before querying live exchange feeds. 'Live (0m)' guarantees zero lag and real-time prices directly from exchange feeds."
)
ttl_map = {"Live (0m)": 0.0, "15m": 0.25, "30m": 0.5, "1h": 1.0, "2h": 2.0, "4h": 4.0}
cache_ttl = ttl_map.get(cache_ttl_option, 0.0)
bypass_cache_flag = force_live_refresh or (cache_ttl == 0.0)

run_btn = st.sidebar.button("🔍 Run Live Market Screen", type="primary", use_container_width=True)

if "data_health" in st.session_state and "refresh_timestamp" in st.session_state["data_health"]:
    st.sidebar.caption(f"🕒 **Last Refreshed:** {st.session_state['data_health']['refresh_timestamp']}")

if is_admin():
    if st.sidebar.button("👑 Admin Flush Cache & Memory", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.toast("✅ Admin memory cache flushed! Re-running screeners...", icon="🧹")
        st.rerun()
else:
    if st.sidebar.button("🧹 Flush Cache & Reset Memory", use_container_width=True):
        st.cache_data.clear()
        st.session_state.clear()
        st.toast("✅ App memory cache flushed! Re-running screeners...", icon="🧹")
        st.rerun()

# ----------------- MAIN APP -----------------
st.title("📈 Indian Equities & Commodities AI Screener")
st.caption("Quantitative Multi-Strategy Algorithmic Screener with Low-Cost AI Trade Plan Generation (NSE, BSE & MCX)")

# Helper to run scan (disk-cached by LocalDataCache)
def execute_screening(u_name, custom_list, strat, top_limit, prov_mode, b_key="", b_sec="", b_tok="", bypass_cache=False, cache_ttl=0.25):
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
    
    # 3. Always compute Indian Market Indices Derivatives Hub & Global Macro Regime
    idx_symbols = ["NIFTY", "BANKNIFTY", "SENSEX", "MIDCPNIFTY", "NIFTYIT", "CRUDEOIL", "GOLD"]
    yf_idx_prov = YahooFinanceProvider(cache_ttl_hours=cache_ttl)
    idx_raw = yf_idx_prov.fetch_batch_ohlcv(idx_symbols, period="6mo", interval="1d", max_workers=5, use_cache=not bypass_cache)
    index_results = IndexDerivativesAnalyzer.analyze_batch(idx_raw)
    try:
        macro_context = MacroMarketEngine.analyze_macro_regime(idx_raw)
    except Exception:
        macro_context = MacroMarketEngine._get_default_macro_dict()

    # 4. Extract data health telemetry from providers
    try:
        p_health = provider.get_health_status() if hasattr(provider, "get_health_status") else {}
    except Exception:
        p_health = {}
    try:
        idx_health = yf_idx_prov.get_health_status() if hasattr(yf_idx_prov, "get_health_status") else {}
    except Exception:
        idx_health = {}

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

    ist_tz = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist_tz)
    refresh_timestamp_str = now_ist.strftime("%I:%M:%S %p IST, %d-%b-%Y")
    refresh_time_short = now_ist.strftime("%I:%M:%S %p IST")

    data_health = {
        "is_latest": is_latest,
        "api_call_failed": api_call_failed,
        "provider_name": prov_mode,
        "data_source_mode": p_health.get("data_source_mode", "LIVE"),
        "failed_symbols": failed_syms,
        "fallback_symbols": fallback_syms,
        "failure_reasons": failure_reasons,
        "latest_data_date": latest_data_date,
        "refresh_timestamp": refresh_timestamp_str,
        "refresh_time_short": refresh_time_short,
        "symbol_timestamps": p_health.get("symbol_timestamps", {}),
        "symbol_sources": p_health.get("symbol_sources", {}),
        "total_symbols_requested": len(symbols),
        "total_symbols_loaded": len(data),
        "is_static_mode": is_static
    }

    # 5. Screen Equities with Benchmark Context & First-to-Recover Dip Leaders
    results = {}
    if strat in ["All Strategies", "First-to-Recover Dip Leaders"]:
        results["DIP_LEADER"] = DipLeaderScreener().screen_batch(data, top_n=top_limit, benchmark_data=idx_raw)
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

    try:
        custom_levels = StockLevelAnalyzer.analyze_batch(data, benchmark_data=idx_raw)
    except Exception:
        try:
            custom_levels = StockLevelAnalyzer.analyze_batch(data, benchmark_data=None)
        except Exception:
            custom_levels = []

    # When custom universe is selected, ensure all custom stocks flow into results for AI Trade Plans
    if u_name == "Custom" and custom_levels:
        custom_cands = []
        for cl in custom_levels:
            custom_cands.append({
                "symbol": cl["symbol"],
                "category": "TWO_WAY_LEVELS",
                "bias": "LONG" if "BULL" in cl["bias"] else ("SHORT" if "BEAR" in cl["bias"] else "LONG"),
                "score": 90.0 if "ACTIVE" in cl["status"] else (75.0 if "TESTING" in cl["status"] else 65.0),
                "close": cl["close"],
                "change_pct": cl["change_pct"],
                "rvol": cl["rvol"],
                "rvol_9": cl["rvol"],
                "vol_confirmed_2x": cl["rvol"] >= 2.0,
                "retest_support_50pct": cl["fall_below"],
                "retracement_healthy": True,
                "sector": cl["sector"],
                "has_sector_tailwind": True,
                "panic_day_resilient": False,
                "mansfield_rs": cl["mansfield_rs"],
                "rsi": cl["rsi"],
                "reasons": [
                    f"🟢 Rise Above: ₹{cl['rise_above']:,.2f} (Upside T1: ₹{cl['upside_target_1']:,.2f})",
                    f"🔴 Fall Below: ₹{cl['fall_below']:,.2f} (Downside T1: ₹{cl['downside_target_1']:,.2f})",
                    f"🟡 Range: {cl['chop_zone']}"
                ],
                "capture_time": cl["capture_time"],
                "data_source": cl["data_source"],
                "rise_above": cl["rise_above"],
                "fall_below": cl["fall_below"],
                "upside_target_1": cl["upside_target_1"],
                "downside_target_1": cl["downside_target_1"],
                "trade_thesis": cl["trade_thesis"],
                "options_setup": cl["options_setup"]
            })
        results["TWO_WAY_LEVELS"] = custom_cands

    elapsed = time.time() - t_start
    return symbols, data, results, index_results, custom_levels, elapsed, data_health, macro_context


if run_btn or "cached_results" in st.session_state:
    if run_btn:
        with st.spinner("Syncing candles, macro indicators & computing vectorized screeners..."):
            symbols, data_dict, results, index_results, custom_levels, elapsed, data_health, macro_context = execute_screening(
                universe_choice,
                custom_tickers,
                strategy_choice,
                top_n,
                provider_choice,
                breeze_api_key_in,
                breeze_secret_key_in,
                breeze_session_token_in,
                bypass_cache=bypass_cache_flag,
                cache_ttl=cache_ttl
            )
            st.session_state["symbols"] = symbols
            st.session_state["data_dict"] = data_dict
            st.session_state["cached_results"] = results
            st.session_state["index_results"] = index_results
            st.session_state["custom_levels"] = custom_levels
            st.session_state["elapsed"] = elapsed
            st.session_state["data_health"] = data_health
            st.session_state["macro_context"] = macro_context

    symbols = st.session_state["symbols"]
    data_dict = st.session_state["data_dict"]
    results = st.session_state["cached_results"]
    index_results = st.session_state.get("index_results", [])
    custom_levels = st.session_state.get("custom_levels", [])
    if not custom_levels and data_dict:
        try:
            custom_levels = StockLevelAnalyzer.analyze_batch(data_dict)
            st.session_state["custom_levels"] = custom_levels
        except Exception:
            pass
    elapsed = st.session_state["elapsed"]
    macro_context = st.session_state.get("macro_context")
    if not macro_context or not isinstance(macro_context, dict):
        try:
            macro_context = MacroMarketEngine.analyze_macro_regime()
        except Exception:
            macro_context = MacroMarketEngine._get_default_macro_dict()
        st.session_state["macro_context"] = macro_context
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
        refresh_ts = data_health.get("refresh_timestamp", "Recent")
        active_banner_html = (
            f'<div style="background: linear-gradient(90deg, #064e3b 0%, #047857 100%); border: 1px solid #10B981; '
            f'border-radius: 8px; padding: 10px 18px; margin-top: 8px; margin-bottom: 18px; display: flex; '
            f'align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">'
            f'<div>'
            f'<div style="color: #ECFDF5; font-size: 0.95rem; font-weight: 700;">'
            f'🟢 <b>Live Data Feed Active</b> — {data_health.get("provider_name", provider_choice)}'
            f'</div>'
            f'<div style="color: #A7F3D0; font-size: 0.82rem; margin-top: 3px;">'
            f'⏰ <b>Source Data Captured / Refreshed at:</b> <span style="background: rgba(0,0,0,0.3); padding: 2px 7px; border-radius: 4px; font-weight: bold; color: #FFF;">{refresh_ts}</span> &nbsp;|&nbsp; Session: <b>{data_health.get("latest_data_date", "Today")}</b>'
            f'</div>'
            f'</div>'
            f'<div style="color: #6EE7B7; font-size: 0.8rem; background: rgba(0,0,0,0.25); padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(110,231,183,0.3);">'
            f'📡 Exchange Sync: <b>Verified Real-Time</b>'
            f'</div>'
            f'</div>'
        )
        render_html(active_banner_html)

    # 1-Click Live Resync Action
    btn_c1, btn_c2 = st.columns([3, 1])
    with btn_c2:
        if st.button("⚡ Force Live Resync", key="btn_dash_live_resync", help="Clears memory and pulls fresh live quotes from exchange feeds immediately", use_container_width=True):
            st.session_state.pop("cached_results", None)
            st.session_state.pop("data_dict", None)
            st.rerun()

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
        "Source Capture Time",
        data_health.get("refresh_time_short", data_health.get("latest_data_date", "N/A")),
        delta=f"Session: {data_health.get('latest_data_date', 'Today')}",
        delta_color="normal" if is_live else "off"
    )

    st.markdown("---")

    levels_tab_name = "🎯 Custom Stock Levels (Rise / Fall)" if universe_choice == "Custom" else "🎯 Two-Way Levels (Rise / Fall)"

    tab_levels, tab_indices, tab_dipleaders, tab_ai, tab_screeners, tab_tracker, tab_charts, tab_docs, tab_audit = st.tabs([
        levels_tab_name,
        "🏛️ Indices & Macros",
        "🚀 First-to-Recover Dip Leaders",
        "🤖 AI Trade Plans",
        "📊 Strategy Shortlists",
        "📌 Live Item Tracker",
        "📈 Technical Chart View",
        "⚙️ Strategy Documentation",
        "🔍 API Data Audit & Proof"
    ])

    with tab_levels:
        st.subheader("🎯 Institutional Two-Way Action Levels (Rise Above / Fall Below)")
        st.caption(
            "Precision quantitative boundary thresholds: Identifies the exact level **ABOVE** which institutional buyers gain control "
            "triggering an upside rally, the level **BELOW** which sellers break support triggering downward acceleration, "
            "the neutral chop zone, and dynamic volatility-based targets / stop loss levels."
        )

        if not custom_levels:
            st.info("ℹ️ No stock levels available. Enter custom stocks in the sidebar or run the screener to calculate two-way breakout and breakdown levels.")
        else:
            # 1. Summary Metrics
            active_rises = sum(1 for x in custom_levels if "EXPANSION" in x["status"] or "RESISTANCE" in x["status"])
            active_falls = sum(1 for x in custom_levels if "BREAKDOWN" in x["status"] or "SUPPORT" in x["status"])
            consolidating = sum(1 for x in custom_levels if "CONSOLIDATING" in x["status"])

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Stocks Evaluated", f"{len(custom_levels)} Assets")
            k2.metric("🟢 Testing / Above Rise Level", f"{active_rises} Setups", delta="Upside Momentum" if active_rises > 0 else "None")
            k3.metric("🔴 Testing / Below Fall Level", f"{active_falls} Setups", delta="Downside Risk" if active_falls > 0 else "None", delta_color="inverse")
            k4.metric("🟡 Consolidating (Chop Zone)", f"{consolidating} Setups")

            st.markdown("---")

            # 2. Executive Two-Way Levels Table
            st.markdown("### 📋 Executive Action Levels Summary Table")
            st.caption("Sort by distance to triggers to see which stocks are immediately pressing key action levels today.")
            
            levels_table = []
            for s in custom_levels:
                levels_table.append({
                    "Symbol": s["symbol"],
                    "Sector": s["sector"],
                    "LTP (₹)": f"₹{s['close']:,.2f}",
                    "Day Change": f"{s['change_pct']:+.2f}%",
                    "🟢 RISE ABOVE": f"₹{s['rise_above']:,.2f}",
                    "Rise Dist %": f"{s['rise_distance_pct']:+.2f}%",
                    "Upside Targets": f"T1: ₹{s['upside_target_1']:,.2f} | T2: ₹{s['upside_target_2']:,.2f}",
                    "Upside SL": f"₹{s['upside_stop_loss']:,.2f} ({s['upside_risk_reward']})",
                    "Call Play (CE)": s["options_setup"]["call_option_play"] if s["options_setup"].get("is_fno") else "N/A (Cash)",
                    "🔴 FALL BELOW": f"₹{s['fall_below']:,.2f}",
                    "Fall Dist %": f"{s['fall_distance_pct']:+.2f}%",
                    "Downside Targets": f"T1: ₹{s['downside_target_1']:,.2f} | T2: ₹{s['downside_target_2']:,.2f}",
                    "Downside SL": f"₹{s['downside_stop_loss']:,.2f} ({s['downside_risk_reward']})",
                    "Put Play (PE)": s["options_setup"]["put_option_play"] if s["options_setup"].get("is_fno") else "N/A (Cash)",
                    "🟡 Chop Zone": s["chop_zone"],
                    "Status": s["status"]
                })
            st.dataframe(pd.DataFrame(levels_table), use_container_width=True, hide_index=True)

            st.markdown("---")

            # 3. Deep-Dive Tactical Execution Cards
            st.markdown("### 🔍 Deep-Dive Stock Execution Cards & 1-Click Tracker")
            for idx_s, s in enumerate(custom_levels):
                sym_name = s["symbol"]
                close_p = s["close"]
                chg = s["change_pct"]
                chg_c = "#00E676" if chg >= 0 else "#FF5252"
                b_color = s["status_color"]
                
                # Card HTML
                card_html = (
                    f'<div style="background-color: #1a1e24; border-radius: 10px; padding: 18px; margin-bottom: 16px; border-left: 6px solid {b_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.35);">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">'
                    f'<div>'
                    f'<h3 style="margin: 0; color: #FFF; font-size: 1.4rem;">{sym_name} <span style="font-size: 0.9rem; color: #94A3B8;">({s["sector"]})</span></h3>'
                    f'<div style="display: flex; align-items: center; gap: 10px; margin-top: 4px; flex-wrap: wrap;">'
                    f'<span style="font-size: 0.95rem; color: #E2E8F0;">LTP: <b style="color: #FFF; font-size: 1.15rem;">₹{close_p:,.2f}</b> (<b style="color: {chg_c};">{chg:+.2f}%</b>)</span>'
                    f'<span style="background: #1e293b; color: #38BDF8; border: 1px solid rgba(56,189,248,0.3); font-size: 0.75rem; padding: 2px 8px; border-radius: 4px;">⏰ {s.get("capture_time", "Live")}</span>'
                    f'<span style="color: #A0AEC0; font-size: 0.8rem;">RSI: <b>{s["rsi"]}</b> | ATR(14): <b>₹{s["atr"]:,.2f}</b> | RVol: <b>{s["rvol"]:.2f}x</b></span>'
                    f'</div>'
                    f'</div>'
                    f'<div style="background-color: #0e1117; border: 1px solid {b_color}; color: {b_color}; padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 0.85rem;">'
                    f'{s["status"]}'
                    f'</div>'
                    f'</div>'
                    f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 14px;">'
                    f'<div style="background-color: #062b1a; border: 1px solid #10B981; border-radius: 8px; padding: 12px 14px;">'
                    f'<div style="color: #34D399; font-weight: bold; font-size: 0.95rem; margin-bottom: 6px;">🟢 RALLY SETUP (BUY ABOVE ₹{s["rise_above"]:,.2f})</div>'
                    f'<div style="color: #E2E8F0; font-size: 0.84rem; line-height: 1.5;">'
                    f'• <b>Breakout Trigger:</b> ₹{s["rise_above"]:,.2f} ({s["rise_distance_pct"]:+.2f}% away)<br>'
                    f'• <b>Target 1 (0.8x ATR):</b> <b style="color: #34D399;">₹{s["upside_target_1"]:,.2f}</b><br>'
                    f'• <b>Target 2 (1.6x ATR):</b> <b style="color: #34D399;">₹{s["upside_target_2"]:,.2f}</b><br>'
                    f'• <b>Target 3 (2.5x ATR):</b> <b style="color: #34D399;">₹{s["upside_target_3"]:,.2f}</b><br>'
                    f'• <b>Stop Loss:</b> <b style="color: #F87171;">₹{s["upside_stop_loss"]:,.2f}</b> (R:R {s["upside_risk_reward"]})<br>'
                    f'• <b>Option Play:</b> <code style="color: #A7F3D0;">{s["options_setup"]["call_option_play"]}</code>'
                    f'</div>'
                    f'</div>'
                    f'<div style="background-color: #3b0d11; border: 1px solid #EF4444; border-radius: 8px; padding: 12px 14px;">'
                    f'<div style="color: #F87171; font-weight: bold; font-size: 0.95rem; margin-bottom: 6px;">🔴 BREAKDOWN SETUP (SELL BELOW ₹{s["fall_below"]:,.2f})</div>'
                    f'<div style="color: #E2E8F0; font-size: 0.84rem; line-height: 1.5;">'
                    f'• <b>Breakdown Floor:</b> ₹{s["fall_below"]:,.2f} ({s["fall_distance_pct"]:+.2f}% away)<br>'
                    f'• <b>Target 1 (0.8x ATR):</b> <b style="color: #F87171;">₹{s["downside_target_1"]:,.2f}</b><br>'
                    f'• <b>Target 2 (1.6x ATR):</b> <b style="color: #F87171;">₹{s["downside_target_2"]:,.2f}</b><br>'
                    f'• <b>Target 3 (2.5x ATR):</b> <b style="color: #F87171;">₹{s["downside_target_3"]:,.2f}</b><br>'
                    f'• <b>Stop Loss:</b> <b style="color: #34D399;">₹{s["downside_stop_loss"]:,.2f}</b> (R:R {s["downside_risk_reward"]})<br>'
                    f'• <b>Option Play:</b> <code style="color: #FECACA;">{s["options_setup"]["put_option_play"]}</code>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="background-color: #131922; border-left: 3px solid #F59E0B; padding: 8px 12px; border-radius: 4px; margin-top: 10px; font-size: 0.85rem; color: #CBD5E0;">'
                    f'<b>🟡 Range / Chop Zone:</b> <code>{s["chop_zone"]}</code> &nbsp;|&nbsp; <i>{s["thesis_neutral"]}</i>'
                    f'</div>'
                    f'</div>'
                )
                render_html(card_html)

                # 1-Click Track Form (Long or Short)
                current_user_name = get_current_user() or "trader"
                with st.expander(f"📌 Track {sym_name} Action Triggers in Live Portfolio"):
                    with st.form(f"track_level_form_{idx_s}_{sym_name}", clear_on_submit=False):
                        tc1, tc2, tc3 = st.columns(3)
                        t_dir = tc1.selectbox("Trade Direction", ["🟢 LONG (Breakout Above)", "🔴 SHORT (Breakdown Below)"], key=f"t_lvl_dir_{idx_s}")
                        is_long = "LONG" in t_dir
                        default_entry = s["rise_above"] if is_long else s["fall_below"]
                        default_sl = s["upside_stop_loss"] if is_long else s["downside_stop_loss"]
                        default_tgt = s["upside_target_1"] if is_long else s["downside_target_1"]
                        
                        t_entry = tc2.number_input("Trigger / Entry Price (₹)", value=float(default_entry), key=f"t_lvl_entry_{idx_s}")
                        t_qty = tc3.number_input("Quantity / Units", value=50.0, min_value=1.0, key=f"t_lvl_qty_{idx_s}")
                        
                        tc4, tc5 = st.columns(2)
                        t_type = tc4.selectbox("Asset Instrument", ["EQUITY", "OPTION_CE", "OPTION_PE", "FUTURES"], index=0 if is_long else 2, key=f"t_lvl_inst_{idx_s}")
                        t_sl = tc5.number_input("Stop Loss (₹)", value=float(default_sl), key=f"t_lvl_sl_{idx_s}")
                        
                        t_btn = st.form_submit_button(f"➕ Add {sym_name} to Tracker", type="primary", use_container_width=True)
                        if t_btn:
                            PortfolioTracker.add_position(
                                username=current_user_name,
                                symbol=sym_name,
                                asset_type=t_type,
                                buy_price=float(t_entry),
                                quantity=float(t_qty),
                                stop_loss=float(t_sl),
                                target=float(default_tgt),
                                notes=f"Action Level setup ({s['status'][:30]})"
                            )
                            st.toast(f"✅ Added {sym_name} ({t_dir[:7]}) to {current_user_name}'s tracker!", icon="📌")
                            time.sleep(0.4)
                            st.rerun()

    with tab_indices:
        st.subheader("🏛️ Indian Market & Sectoral Indices Derivatives Hub")
        st.caption("Quantitative Futures signals (Buy / Sell / Hold), Option Strategies (CE/PE/Spreads), and Profitability Ranking for NIFTY 50, BANK NIFTY, SENSEX, MIDCAP NIFTY, and Sectoral Indices.")

        # Broader Market & Macro Cross-Asset Pulse
        m_nifty = macro_context.get("nifty_summary", {})
        m_crude = macro_context.get("crude_summary", {})
        m_vix = macro_context.get("vix_summary", {})
        m_usdinr = macro_context.get("usdinr_summary", {})

        with st.container():
            st.markdown("#### 🌍 Broader Market & Macro Cross-Asset Pulse")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                n_chg = m_nifty.get("change_pct", 0.0)
                st.metric("NIFTY 50", f"₹{m_nifty.get('close', 22600.0):,.1f}", delta=f"{n_chg:+.2f}%", delta_color="normal" if n_chg >= 0 else "inverse")
            with mc2:
                c_price = m_crude.get("price_usd", 94.0)
                st.metric("Crude Oil", f"${c_price:.2f}/bbl", delta=f"{m_crude.get('change_pct', 0.0):+.2f}%", delta_color="inverse" if c_price > 85 else "normal")
            with mc3:
                u_val = m_usdinr.get("rate", 96.0)
                st.metric("USD / INR", f"₹{u_val:.2f}", delta="Exporter Tailwind" if u_val >= 83.5 else "Neutral", delta_color="normal")
            with mc4:
                v_val = m_vix.get("level", 14.0)
                st.metric("India VIX", f"{v_val:.2f}", delta=m_vix.get('regime', 'NORMAL').replace('_', ' '), delta_color="inverse" if v_val > 18 else "normal")
            st.markdown(f"<div style='background-color: #1e293b; border-left: 4px solid #38BDF8; padding: 6px 12px; border-radius: 4px; font-size: 0.84rem; color: #E2E8F0; margin-bottom: 16px;'><b>Macro Environment:</b> {macro_context.get('macro_regime_badge', '🟢 RISK-ON')} | <b>Playbook:</b> {macro_context.get('trading_playbook', 'Trade quality setups.')}</div>", unsafe_allow_html=True)

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

    with tab_dipleaders:
        st.subheader("🚀 First-to-Recover Dip Leaders (The Coiled Spring Strategy)")
        st.caption("Quantitative detection of Stage-2 market leaders that pulled back alongside broader market corrections on low/drying volume. Backed by institutional accumulation and moving average defense, these resilient stocks are primed to rebound first and fastest when NIFTY/BANKNIFTY turns up.")

        # 1. Real-Time Macro Market & Commodity Cross-Asset Indicator Ribbon
        m_nifty = macro_context.get("nifty_summary", {})
        m_crude = macro_context.get("crude_summary", {})
        m_vix = macro_context.get("vix_summary", {})
        m_usdinr = macro_context.get("usdinr_summary", {})

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            n_chg = m_nifty.get("change_pct", 0.0)
            st.metric(
                "NIFTY 50 (Benchmark)",
                f"₹{m_nifty.get('close', 22600.0):,.1f}",
                delta=f"{n_chg:+.2f}% ({m_nifty.get('trend', 'BULLISH')})",
                delta_color="normal" if n_chg >= 0 else "inverse"
            )
            st.caption(f"RSI: **{m_nifty.get('rsi', 50.0)}** | Trend: **{m_nifty.get('trend', 'BULLISH')}**")

        with m_col2:
            c_usd = m_crude.get("price_usd", 94.0)
            c_chg = m_crude.get("change_pct", 0.0)
            c_mcx = m_crude.get("price_mcx_inr", 0.0)
            c_lbl = f"${c_usd:.2f}/bbl"
            if c_mcx > 0:
                c_lbl += f" (₹{c_mcx:,.0f} MCX)"
            st.metric(
                "Crude Oil (Macro Input)",
                c_lbl,
                delta=f"{c_chg:+.2f}% ({m_crude.get('trend', 'NEUTRAL')})",
                delta_color="inverse" if c_usd > 85 else "normal"
            )
            st.caption(f"Consumers: **{m_crude.get('impact_consumers', 'NEUTRAL')}** | Upstream: **{m_crude.get('impact_upstream', 'NEUTRAL')}**")

        with m_col3:
            u_rate = m_usdinr.get("rate", 96.0)
            st.metric(
                "USD/INR (Currency)",
                f"₹{u_rate:.2f} / $",
                delta="Exporter Tailwind" if u_rate >= 83.5 else "Stable",
                delta_color="normal"
            )
            st.caption("IT & Pharma Exporter Revenue Tailwind: **POSITIVE**")

        with m_col4:
            v_lvl = m_vix.get("level", 14.0)
            v_regime = m_vix.get("regime", "NORMAL_VOLATILITY")
            st.metric(
                "India VIX (Risk Gauge)",
                f"{v_lvl:.2f}",
                delta=v_regime.replace("_", " "),
                delta_color="inverse" if v_lvl > 18 else "normal"
            )
            st.caption(f"Market Volatility Risk: **{m_vix.get('risk_level', 'MODERATE')}**")

        # Macro Regime Status Banner
        regime_badge = macro_context.get("macro_regime_badge", "🟢 RISK-ON / MACRO TAILWIND")
        playbook = macro_context.get("trading_playbook", "Trade breakouts and dip leaders with full conviction.")
        raw_score = macro_context.get("macro_score", 40)
        try:
            m_score_val = int(round(float(raw_score)))
            m_score_str = f"{m_score_val:+d}"
        except (ValueError, TypeError):
            m_score_str = "+0"
        
        banner_bg = "linear-gradient(90deg, #064e3b 0%, #047857 100%)" if "RISK-ON" in regime_badge or "DIP" in regime_badge else ("linear-gradient(90deg, #78350f 0%, #b45309 100%)" if "CAUTIOUS" in regime_badge else "linear-gradient(90deg, #450a0a 0%, #7f1d1d 100%)")
        banner_border = "#10B981" if "RISK-ON" in regime_badge or "DIP" in regime_badge else ("#F59E0B" if "CAUTIOUS" in regime_badge else "#EF4444")
        
        macro_html = (
            f'<div style="background: {banner_bg}; border: 1px solid {banner_border}; border-radius: 8px; padding: 14px 20px; margin: 12px 0 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">'
            f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">'
            f'<div>'
            f'<span style="background-color: {banner_border}; color: #000; font-weight: 800; padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; letter-spacing: 0.5px;">MACRO REGIME AUDIT</span>'
            f'<h3 style="margin: 6px 0 2px 0; color: #FFF; font-size: 1.3rem;">{regime_badge} <span style="font-size: 1rem; color: #CBD5E0;">(Macro Score: {m_score_str}/100)</span></h3>'
            f'<div style="color: #F1F5F9; font-size: 0.9rem; line-height: 1.4;"><b>Institutional Playbook:</b> {playbook}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        render_html(macro_html)

        # 2. Educational Philosophy Expander
        with st.expander("💡 The Quantitative Coiled Spring Philosophy — How We Identify First-to-Recover Stocks", expanded=False):
            st.markdown(r"""
            ### 🎯 Why Do These Specific Stocks Recover First & Fastest?
            When broader markets (NIFTY 50 / BANK NIFTY) face selling waves or macro panics (e.g., crude spikes or geopolitical fears), retail traders panic and sell everything indiscriminately.
            However, **true market leaders** are characterized by five rigorous quantitative hallmarks:

            1. **Stage-2 Structural Dominance**:
               The stock is in an established primary uptrend ($\text{Price} > \text{SMA}_{200}$ and $\text{EMA}_{20} > \text{EMA}_{50}$). It was leading before the dip and will lead after.
            2. **Panic-Day Resilience & Dip Alpha**:
               During sessions where NIFTY tumbled $\le -0.75\%$, these stocks showed positive relative strength ($\text{Stock Return} - \text{NIFTY Return} \ge +1.0\%$) or even closed green. Institutional hands actively soaked up retail panic supply.
            3. **High Up-Beta (Rebound Elasticity)**:
               $$\text{Up-Beta} = \frac{\text{Mean}(\text{Stock Return} \mid \text{NIFTY Return} > 0)}{\text{Mean}(\text{NIFTY Return} \mid \text{NIFTY Return} > 0)}$$
               Stocks with Up-Beta $\ge 1.25\text{x}$ act like coiled springs. The moment NIFTY gains $+0.8\%$, these leaders explode $+2.0\%$ to $+4.5\%$.
            4. **Volume Exhaustion on Pullbacks (Dry Supply)**:
               While price pulls back, trading volume **dries up** ($\text{RVol}_{20d} < 1.10\text{x}$). Smart money is **not** selling; only weak retail liquidity is trickling out.
            5. **Dynamic Floor Defense with Lower Wicks**:
               Price pulls back directly into key institutional floors (20 EMA, 50 EMA, or 50% breakout candle body midpoint) and leaves a prominent lower wick ($\ge 25\%$ of candle range), showing buyer absorption at the floor.
            """)

        # 3. Dip Leader Candidates Grid
        dip_candidates = results.get("DIP_LEADER", [])
        if not dip_candidates:
            st.info("ℹ️ **No Dip Leader candidates detected in the current universe selection.** This usually occurs when the selected universe is in a clean parabolic breakout or when stocks have not recently dipped to test their 20/50 EMA support floors. Try switching the Universe filter to **Nifty 50** or **Custom** with `SUNPHARMA, CIPLA, DIVISLAB, HDFCBANK, RELIANCE` and rerun the screen.")
        else:
            st.markdown(f"### 💎 Identified First-to-Recover Leaders ({len(dip_candidates)} Setups)")
            
            d_cols = st.columns(2)
            for d_idx, c in enumerate(dip_candidates):
                col = d_cols[d_idx % 2]
                with col:
                    d_sym = c.get("symbol", "")
                    d_chg = c.get("change_pct", 0.0)
                    chg_color = "#00E676" if d_chg >= 0 else "#FF5252"
                    
                    v_badge = c.get("recovery_velocity_badge", "🚀 COILED SPRING (HIGH BOUNCE VELOCITY)")
                    v_border = "#00E676" if "COILED" in v_badge else "#64B5F6"
                    
                    # Macro alignment
                    macro_align_html = ""
                    if c.get("macro_alignment"):
                        macro_align_html = f'<div style="background-color: #0e2a3b; border-left: 3px solid #38BDF8; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; color: #BAE6FD; margin: 6px 0;"><b>🌊 Macro Alignment:</b> {c.get("macro_alignment")}</div>'
                    
                    # Card HTML
                    card_html = (
                        f'<div style="background-color: #1a1e24; border-radius: 10px; padding: 18px; margin-bottom: 16px; border-left: 6px solid {v_border}; box-shadow: 0 4px 10px rgba(0,0,0,0.35);">'
                        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                        f'<div>'
                        f'<h3 style="margin: 0; color: #FFF; font-size: 1.35rem;">{d_sym} <span style="font-size: 0.85rem; color: #94A3B8;">({c.get("sector", "Broad Market")})</span></h3>'
                        f'<div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 3px;">'
                        f'<span style="font-size: 0.88rem; color: #CBD5E0;">LTP: <b style="color: #FFF; font-size: 1.05rem;">₹{c.get("close", 0.0):,.2f}</b> (<b style="color: {chg_color};">{d_chg:+.2f}%</b>)</span>'
                        f'<span style="background: #1e293b; color: #38BDF8; border: 1px solid rgba(56,189,248,0.3); font-size: 0.72rem; padding: 2px 7px; border-radius: 4px;">⏰ Captured: {c.get("capture_time", data_health.get("refresh_time_short", "Live"))}</span>'
                        f'</div>'
                        f'</div>'
                        f'<span style="background-color: #064e3b; color: #34D399; border: 1px solid #10B981; padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">Score: {c.get("score")}/100</span>'
                        f'</div>'
                        f'<div style="margin: 8px 0 6px 0;">'
                        f'<span style="background-color: #1e293b; color: #F1F5F9; border-left: 3px solid {v_border}; padding: 3px 8px; border-radius: 3px; font-size: 0.8rem; font-weight: 600;">{v_badge}</span>'
                        f'</div>'
                        f'{macro_align_html}'
                        f'<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; background-color: #0e1117; padding: 10px; border-radius: 6px; margin: 10px 0 8px 0;">'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Dip Alpha vs Nifty</span><br><b style="color: #00E676; font-size: 1.0rem;">+{c.get("dip_alpha_pct", 0.0):.2f}%</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Up-Beta (Bounce)</span><br><b style="color: #FFD54F; font-size: 1.0rem;">{c.get("bounce_velocity_up_beta", 1.0):.2f}x</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Pullback Volume</span><br><b style="color: #64B5F6; font-size: 1.0rem;">{c.get("pullback_rvol", 0.8):.2f}x (Dry)</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Support Floor Held</span><br><b style="color: #FFF; font-size: 0.88rem;">{c.get("support_floor_desc", "20 EMA")}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Buyer Wick Defense</span><br><b style="color: #A7F3D0; font-size: 0.88rem;">{c.get("lower_wick_pct", 0.0):.1f}% Wick</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">MRS vs Nifty (50)</span><br><b style="color: #F472B6; font-size: 0.88rem;">{c.get("mrs", 0.0):+.2f}%</b></div>'
                        f'</div>'
                        f'<div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; background-color: #131922; padding: 10px; border-radius: 6px; margin-bottom: 8px;">'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Springboard Entry</span><br><b style="color: #FFF; font-size: 0.95rem;">₹{c.get("springboard_entry_zone", 0.0):,.1f}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Invalidation SL</span><br><b style="color: #FF5252; font-size: 0.95rem;">₹{c.get("invalidation_sl", 0.0):,.1f}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Rebound Target 1</span><br><b style="color: #00E676; font-size: 0.95rem;">₹{c.get("rebound_target_1", 0.0):,.1f}</b></div>'
                        f'<div><span style="color: #888; font-size: 0.72rem;">Rebound Target 2</span><br><b style="color: #00E676; font-size: 0.95rem;">₹{c.get("rebound_target_2", 0.0):,.1f}</b></div>'
                        f'</div>'
                        f'<div style="margin: 8px 0 0 0; color: #E2E8F0; font-size: 0.85rem; line-height: 1.45;"><i>&ldquo;{c.get("recovery_catalyst", "")}&rdquo;</i></div>'
                        f'</div>'
                    )
                    render_html(card_html)

                    # 1-Click Track in Live Item Tracker
                    current_user_name = get_current_user() or "trader"
                    with st.expander(f"📌 Track {d_sym} Rebound in My Item Tracker"):
                        with st.form(f"track_dip_form_{d_idx}_{d_sym}", clear_on_submit=False):
                            tc1, tc2 = st.columns(2)
                            t_entry = tc1.number_input("Entry Price (₹)", value=float(c.get("springboard_entry_zone", c.get("close", 100.0))), key=f"t_dip_buy_{d_idx}")
                            t_qty = tc2.number_input("Quantity / Units", value=50.0, min_value=1.0, key=f"t_dip_qty_{d_idx}")
                            
                            tc3, tc4 = st.columns(2)
                            t_type = tc3.selectbox("Asset Type", ["EQUITY", "OPTION_CE", "FUTURES"], index=0, key=f"t_dip_type_{d_idx}")
                            t_sl = tc4.number_input("Invalidation Stop Loss (₹)", value=float(c.get("invalidation_sl", t_entry * 0.97)), key=f"t_dip_sl_{d_idx}")
                            
                            t_tgt = float(c.get("rebound_target_1", t_entry * 1.08))
                            t_notes = f"Dip Leader setup ({c.get('recovery_velocity_badge', '')[:30]})"
                            
                            t_sub = st.form_submit_button(f"➕ Track {d_sym} in Live Portfolio", type="primary", use_container_width=True)
                            if t_sub:
                                PortfolioTracker.add_position(
                                    username=current_user_name,
                                    symbol=d_sym,
                                    asset_type=t_type,
                                    buy_price=float(t_entry),
                                    quantity=float(t_qty),
                                    stop_loss=float(t_sl),
                                    target=float(t_tgt),
                                    notes=t_notes
                                )
                                st.toast(f"✅ Added {d_sym} to {current_user_name}'s Item Tracker!", icon="📌")
                                time.sleep(0.4)
                                st.rerun()

            # 4. Interactive Summary Table
            st.markdown("### 📋 Dip Leaders Quantitative Comparison Table")
            dip_tbl = []
            for c in dip_candidates:
                dip_tbl.append({
                    "Symbol": c.get("symbol"),
                    "Score": c.get("score"),
                    "LTP (₹)": f"₹{c.get('close', 0.0):,.2f}",
                    "Change %": f"{c.get('change_pct', 0.0):+.2f}%",
                    "Dip Alpha %": f"{c.get('dip_alpha_pct', 0.0):+.2f}%",
                    "Up-Beta": f"{c.get('bounce_velocity_up_beta', 1.0):.2f}x",
                    "Support Floor": c.get("support_floor_desc", "20 EMA"),
                    "Pullback RVol": f"{c.get('pullback_rvol', 1.0):.2f}x",
                    "Lower Wick %": f"{c.get('lower_wick_pct', 0.0):.1f}%",
                    "Springboard Entry": f"₹{c.get('springboard_entry_zone', 0.0):,.1f}",
                    "Invalidation SL": f"₹{c.get('invalidation_sl', 0.0):,.1f}",
                    "Target 1": f"₹{c.get('rebound_target_1', 0.0):,.1f}",
                    "Target 2": f"₹{c.get('rebound_target_2', 0.0):,.1f}",
                    "Sector": c.get("sector", "Broad"),
                    "Macro Alignment": c.get("macro_alignment", "Neutral")
                })
            st.dataframe(pd.DataFrame(dip_tbl), use_container_width=True)

            # 5. Technical Candlestick Inspector for Dip Leaders
            st.markdown("### 🔍 Technical Rebound Candlestick Inspector")
            d_syms_chart = [c["symbol"] for c in dip_candidates if c["symbol"] in data_dict]
            if d_syms_chart:
                d_sel_sym = st.selectbox("Select Dip Leader to Inspect Support Floor & Wicks", d_syms_chart, key="sel_dip_chart")
                if d_sel_sym in data_dict:
                    df_d_chart = enrich_with_indicators(data_dict[d_sel_sym].tail(90).copy())
                    
                    fig_d = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=(f"{d_sel_sym} — Dynamic Support Defense & Coiled Spring Floor", "Volume & Selling Exhaustion"),
                        row_width=[0.25, 0.75]
                    )
                    
                    fig_d.add_trace(go.Candlestick(
                        x=df_d_chart.index,
                        open=df_d_chart['open'],
                        high=df_d_chart['high'],
                        low=df_d_chart['low'],
                        close=df_d_chart['close'],
                        name='Price'
                    ), row=1, col=1)

                    if 'ema_20' in df_d_chart.columns:
                        fig_d.add_trace(go.Scatter(x=df_d_chart.index, y=df_d_chart['ema_20'], line=dict(color='#00E676', width=1.5), name='EMA 20 (Fast Floor)'), row=1, col=1)
                    if 'ema_50' in df_d_chart.columns:
                        fig_d.add_trace(go.Scatter(x=df_d_chart.index, y=df_d_chart['ema_50'], line=dict(color='#29B6F6', width=1.5), name='EMA 50 (Structural Floor)'), row=1, col=1)
                    if 'sma_200' in df_d_chart.columns:
                        fig_d.add_trace(go.Scatter(x=df_d_chart.index, y=df_d_chart['sma_200'], line=dict(color='#AB47BC', width=1.5), name='SMA 200 (Stage-2 Filter)'), row=1, col=1)

                    cand_obj = next((c for c in dip_candidates if c["symbol"] == d_sel_sym), None)
                    if cand_obj and cand_obj.get("invalidation_sl"):
                        fig_d.add_hline(
                            y=cand_obj["invalidation_sl"],
                            line_dash="dash",
                            line_color="#FF5252",
                            annotation_text=f"Invalidation SL (₹{cand_obj['invalidation_sl']:,.1f})",
                            annotation_position="bottom right",
                            row=1, col=1
                        )
                    if cand_obj and cand_obj.get("rebound_target_1"):
                        fig_d.add_hline(
                            y=cand_obj["rebound_target_1"],
                            line_dash="dash",
                            line_color="#00E676",
                            annotation_text=f"Rebound T1 (₹{cand_obj['rebound_target_1']:,.1f})",
                            annotation_position="top right",
                            row=1, col=1
                        )

                    v_colors = ['#FF5252' if c < o else '#00E676' for c, o in zip(df_d_chart['close'], df_d_chart['open'])]
                    fig_d.add_trace(go.Bar(
                        x=df_d_chart.index,
                        y=df_d_chart['volume'],
                        marker_color=v_colors,
                        name='Volume'
                    ), row=2, col=1)

                    fig_d.update_layout(
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=550,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_d, use_container_width=True)

    with tab_ai:
        st.subheader("Actionable Trade Plans & Risk Management")
        if not all_candidates:
            st.info("No candidates met the strict screening criteria today. Markets may be consolidating.")
        else:
            advisor = AIAdvisor(api_key=gemini_key_input if gemini_key_input else None)
            engine_name = "Google Gemini Flash (Live LLM)" if advisor.is_ai_ready else "Algorithmic Precision Engine"
            st.caption(f"Trade plans generated via **{engine_name}** | Broader Macro Regime: **{macro_context.get('macro_regime_badge', '🟢 RISK-ON')}**")

            with st.spinner("Generating risk-reward evaluations, macro impact and trade thesis..."):
                plans = advisor.analyze_candidates(all_candidates, macro_context=macro_context)

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

                    macro_b = p.get("macro_badge", "")
                    macro_badge_html = f' | <b style="color: #64B5F6;">{macro_b}</b>' if macro_b else ''

                    # Short / Put option explainer banner
                    short_banner = ""
                    if not is_bullish:
                        short_banner = (
                            f'<div style="background-color: #2b0d10; border-left: 4px solid #FF5252; padding: 6px 12px; border-radius: 4px; font-size: 0.8rem; color: #FFCDD2; margin-bottom: 8px;">'
                            f'<b>📉 Short / Bearish Setup:</b> Profit is booked as price declines below Entry (₹{p.get("entry_price")}) toward Downside Targets.'
                            f'</div>'
                        )

                    macro_impact_banner = ""
                    if macro_b:
                        is_tailwind = "TAILWIND" in macro_b.upper() or "BENEFICIARY" in macro_b.upper()
                        mb_color = "#10B981" if is_tailwind else "#F59E0B"
                        mb_bg = "#064e3b33" if is_tailwind else "#78350f33"
                        macro_impact_banner = (
                            f'<div style="background-color: {mb_bg}; border-left: 4px solid {mb_color}; padding: 6px 12px; border-radius: 4px; font-size: 0.82rem; color: #E2E8F0; margin-bottom: 8px;">'
                            f'<b>🌍 Macro Factor Impact:</b> {macro_b}'
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
                        f'Category: <b style="color: #64B5F6;">{p.get("category")}</b>{unit_badge}{macro_badge_html} | '
                        f'Timeframe: <b style="color: #FFF;">{p.get("timeframe")}</b> | '
                        f'Conviction: <b style="color: #FFD54F;">{p.get("conviction")}</b>'
                        f'</p>'
                        f'{short_banner}'
                        f'{macro_impact_banner}'
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
                        "Quote Captured": c.get("capture_time", data_health.get("refresh_time_short", "Live")),
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
                    if "dip_alpha_pct" in c:
                        row["Dip Alpha %"] = f"{c['dip_alpha_pct']:+.2f}%"
                    if "bounce_velocity_up_beta" in c:
                        row["Up-Beta"] = f"{c['bounce_velocity_up_beta']:.2f}x"
                    if "springboard_entry_zone" in c:
                        row["Springboard Entry"] = f"₹{c['springboard_entry_zone']:,.1f}"
                    if "invalidation_sl" in c:
                        row["Invalidation SL"] = f"₹{c['invalidation_sl']:,.1f}"
                    if "rebound_target_1" in c:
                        row["Rebound T1"] = f"₹{c['rebound_target_1']:,.1f}"
                    table_data.append(row)
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
            else:
                st.caption("No stocks met this specific category's thresholds.")

    with tab_tracker:
        render_item_tracker_view(current_user=get_current_user() or "trader", data_dict=data_dict)

    with tab_charts:
        st.subheader("Interactive Candlestick & Technical Inspector")
        all_chart_syms = list(dict.fromkeys(
            [c["symbol"] for c in custom_levels] + 
            [c["symbol"] for c in all_candidates] + 
            list(data_dict.keys())
        ))
        if all_chart_syms:
            selected_sym = st.selectbox("Select Candidate / Stock to Inspect", all_chart_syms)
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
                    subplot_titles=(f"{selected_sym} — Two-Way Action Levels & Price Action{comm_chart_tag}", "Volume & RVol"),
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

                # Two-Way Action Levels Overlay (Rise Above / Fall Below / Chop Zone)
                cand_levels = next((x for x in custom_levels if x["symbol"] == selected_sym), None)
                if not cand_levels and selected_sym in data_dict:
                    try:
                        cand_levels = StockLevelAnalyzer.analyze_stock(selected_sym, data_dict[selected_sym])
                    except Exception:
                        cand_levels = None

                if cand_levels:
                    # Highlight Neutral Chop Range
                    fig.add_hrect(
                        y0=cand_levels['fall_below'],
                        y1=cand_levels['rise_above'],
                        fillcolor="rgba(255, 213, 79, 0.08)",
                        line_width=1,
                        line_color="rgba(255, 213, 79, 0.25)",
                        annotation_text=f"Chop Zone (₹{cand_levels['fall_below']:,.1f} - ₹{cand_levels['rise_above']:,.1f})",
                        annotation_position="top left",
                        row=1, col=1
                    )
                    # Green line for Rise Above (Upside Trigger)
                    fig.add_hline(
                        y=cand_levels['rise_above'],
                        line_dash="dash",
                        line_color="#00E676",
                        annotation_text=f"🟢 Rise Above: ₹{cand_levels['rise_above']:,.1f}",
                        annotation_position="top right",
                        row=1, col=1
                    )
                    # Red line for Fall Below (Downside Trigger)
                    fig.add_hline(
                        y=cand_levels['fall_below'],
                        line_dash="dash",
                        line_color="#FF5252",
                        annotation_text=f"🔴 Fall Below: ₹{cand_levels['fall_below']:,.1f}",
                        annotation_position="bottom right",
                        row=1, col=1
                    )
                    # Upside T1
                    if cand_levels.get('upside_target_1'):
                        fig.add_hline(
                            y=cand_levels['upside_target_1'],
                            line_dash="dot",
                            line_color="#69F0AE",
                            annotation_text=f"Upside T1 (₹{cand_levels['upside_target_1']:,.1f})",
                            annotation_position="top right",
                            row=1, col=1
                        )
                    # Downside T1
                    if cand_levels.get('downside_target_1'):
                        fig.add_hline(
                            y=cand_levels['downside_target_1'],
                            line_dash="dot",
                            line_color="#FF8A80",
                            annotation_text=f"Downside T1 (₹{cand_levels['downside_target_1']:,.1f})",
                            annotation_position="bottom right",
                            row=1, col=1
                        )

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

        7. **First-to-Recover Dip Leaders (The Coiled Spring Strategy)**:
           - **Stage-2 Structural Trend**: $\text{Price} > \text{SMA}_{200}$ and $\text{EMA}_{20} > \text{EMA}_{50}$. Ensures the stock was an institutional leader before market weakness and remains in a structural bull cycle.
           - **Panic-Day Dip Alpha vs NIFTY**: Measures relative performance during market drop sessions ($\text{NIFTY} \le -0.75\%$). Dip leaders display $\text{Stock Return} - \text{NIFTY Return} \ge +1.0\%$ or close green, showing smart money absorption of panic selling.
           - **High Up-Beta (Bounce Elasticity)**: Measures upside response when NIFTY turns green. Leaders typically have $\text{Up-Beta} \ge 1.25\text{x}$, delivering rapid multi-fold upside as soon as market selling pressure eases.
           - **Volume Supply Exhaustion**: During pullbacks into support, trading volume drops ($\text{RVol}_{20d} < 1.10\text{x}$). Confirms absence of institutional distribution.
           - **Dynamic Moving Average Support Defense**: Price pulls back to test 20 EMA, 50 EMA, or 50% candle body support and leaves a prominent lower wick ($\ge 25\%$ of candle range), signaling immediate buyer demand.

        8. **Macroeconomic & Commodity Cross-Asset Integration**:
           - **Crude Oil ($/bbl & MCX ₹/bbl)**:
             - *Headwinds*: Elevated crude ($> \$85/\text{bbl}$) compresses gross margins for consumer sectors: Paints (`ASIANPAINT`, `BERGEPAINT`), Tyres (`APOLLOTYRE`, `MRF`), Aviation (`INDIGO`), OMCs (`BPCL`, `HPCL`), and Auto (`MARUTI`).
             - *Tailwinds*: Upstream exploration (`ONGC`, `OIL`) directly expands realizations.
           - **USD/INR Exchange Rate**:
             - Rupee depreciation expands operating profit margins and realizations for IT exporters (`TCS`, `INFY`, `HCLTECH`, `LTIM`) and Pharma exporters (`SUNPHARMA`, `DRREDDY`, `CIPLA`, `DIVISLAB`).
           - **India VIX Volatility Filter**:
             - High VIX ($> 18$) mandates tight risk controls, wider trailing playbooks, and favors defensive dip leaders over speculative high-beta breakouts.
        """)

    with tab_audit:
        st.subheader("🔍 API Raw Data & Indicator Proof (Admin Audit)")
        st.caption("Inspect exact timestamps, OHLCV candles, and computed mathematical formulas to cross-verify against NSE/MCX.")
        
        if is_admin():
            st.markdown(
                '<div style="background: rgba(255, 215, 0, 0.1); border-left: 4px solid #FFD700; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; color: #FFF8E1; font-size: 0.88rem;">'
                '👑 <b>Administrator Audit Clearance:</b> You have full administrative access to review raw exchange candlesticks, diagnostic logs, and export system audit dumps.'
                '</div>',
                unsafe_allow_html=True
            )
        
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
        avail_audit_syms = sorted(list(data_dict.keys())) if data_dict else [c['symbol'] for c in all_candidates]
        if avail_audit_syms:
            inspect_sym = st.selectbox("Select Asset to Audit", avail_audit_syms, key="audit_sym_picker")
            if inspect_sym in data_dict:
                raw_df = data_dict[inspect_sym]
                enriched_df = enrich_with_indicators(raw_df.copy())
                last_candle = enriched_df.iloc[-1]
                
                c_time = raw_df.attrs.get("capture_time", data_health.get("refresh_timestamp", "Recent"))
                d_src = raw_df.attrs.get("data_source", data_health.get("provider_name", "Live Market Feed"))
                last_candle_dt = last_candle.name.strftime('%d-%b-%Y') if hasattr(last_candle.name, 'strftime') else str(last_candle.name)[:10]

                st.markdown(
                    f"<div style='background-color: #1e293b; border-left: 4px solid #38BDF8; padding: 10px 14px; border-radius: 6px; margin: 8px 0 14px 0; color: #E2E8F0; font-size: 0.9rem;'>"
                    f"🕒 <b>Source Capture Timestamp:</b> <code style='color: #38BDF8; font-weight: bold;'>{c_time}</code> &nbsp;|&nbsp; "
                    f"📡 <b>Data Feed:</b> <b>{d_src}</b> &nbsp;|&nbsp; "
                    f"📅 <b>Latest Candlestick Session:</b> <b>{last_candle_dt}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Close Price (LTP)", f"₹{last_candle['close']:.2f}", delta=c_time)
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
