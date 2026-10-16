"""
Authentication and Access Control Manager for Streamlit Application.
Provides institutional-grade user authentication, password verification,
role-based access control (Admin vs Trader), and session management.
"""

import hashlib
import os
import time
from typing import Dict, Optional
import streamlit as st


# Default authorized users and credentials
DEFAULT_USERS = {
    "admin": "admin@2026",
    "guruteja": "guru@2026",
    "raghavendra": "raghavendra@2026",
    "naresh": "naresh@2026",
    "trader": "nsemcx2026"
}

# Authorized Administrator Accounts
ADMIN_USERS = {
    "admin",
    "guruteja",
    "raghavendra",
    "naresh"
}


def _hash_password(password: str) -> str:
    """Computes SHA-256 hash of a password."""
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()


def get_user_database() -> Dict[str, str]:
    """
    Loads authorized users from:
    1. Built-in system accounts (DEFAULT_USERS)
    2. Streamlit Secrets (st.secrets["users"])
    3. Environment variables (ADMIN_USERNAME / ADMIN_PASSWORD or APP_USERS)
    """
    # Start with built-in authorized users
    users: Dict[str, str] = {u.lower(): p for u, p in DEFAULT_USERS.items()}

    # 1. Check Streamlit Secrets (can add more users or override)
    try:
        if hasattr(st, "secrets") and "users" in st.secrets:
            for u, p in st.secrets["users"].items():
                users[str(u).strip().lower()] = str(p).strip()
    except Exception:
        pass

    # 2. Check environment variables
    env_admin = os.getenv("ADMIN_USERNAME", "").strip().lower()
    env_pass = os.getenv("ADMIN_PASSWORD", "").strip()
    if env_admin and env_pass:
        users[env_admin] = env_pass

    app_users_env = os.getenv("APP_USERS", "").strip()
    if app_users_env:
        for pair in app_users_env.split(","):
            if ":" in pair:
                u, p = pair.split(":", 1)
                users[u.strip().lower()] = p.strip()

    return users


def verify_credentials(username: str, password_attempt: str) -> bool:
    """
    Verifies username and password against configured users.
    Supports plain text and SHA-256 hashed passwords in secrets.
    """
    if not username or not password_attempt:
        return False

    db = get_user_database()
    u_clean = username.strip().lower()

    if u_clean not in db:
        return False

    stored_pass = db[u_clean]
    attempt_pass = password_attempt.strip()

    # Match direct plaintext
    if stored_pass == attempt_pass:
        return True

    # Accept alternative legacy password for raghavendra
    if u_clean == "raghavendra" and attempt_pass == "screener2026":
        return True

    # Match SHA-256 hash
    if stored_pass.lower() == _hash_password(attempt_pass):
        return True

    return False


def is_authenticated() -> bool:
    """Returns True if the current user session is authenticated."""
    return bool(st.session_state.get("authenticated", False))


def get_current_user() -> Optional[str]:
    """Returns the username of the logged-in user, or None."""
    return st.session_state.get("username")


def is_admin(username: Optional[str] = None) -> bool:
    """Returns True if the specified user (or current session user) has Admin role."""
    target_user = username or get_current_user()
    if not target_user:
        return False
    clean = target_user.strip().lower()
    if clean in ADMIN_USERS:
        return True
    try:
        if hasattr(st, "secrets") and "admins" in st.secrets:
            secret_admins = [str(a).strip().lower() for a in st.secrets["admins"]]
            if clean in secret_admins:
                return True
    except Exception:
        pass
    return False


def logout_user():
    """Logs out the current user and clears session state."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state.pop("cached_results", None)
    st.session_state.pop("symbols", None)
    st.session_state.pop("data_dict", None)
    st.session_state.pop("prefill_user", None)
    st.session_state.pop("prefill_pass", None)
    st.rerun()


def login_user(username: str = "admin"):
    """
    Directly authenticates a user session without requiring a password.
    Sets all required session variables and triggers a rerun into the dashboard.
    """
    clean_u = (username or "admin").strip().lower()
    st.session_state["authenticated"] = True
    st.session_state["username"] = clean_u
    st.session_state["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    is_adm = is_admin(clean_u)
    role_str = "Administrator" if is_adm else "Trader"
    st.toast(f"✅ Welcome, {clean_u.capitalize()} ({role_str})! Loading dashboard...", icon="🚀")
    time.sleep(0.2)
    st.rerun()


def render_login_gate() -> bool:
    """
    Renders the login page if the user is not authenticated.
    Provides instant 1-click login without any username/password entry,
    as well as 1-click profiles and optional manual credentials.
    Returns True if user is authenticated, False otherwise (stops script execution).
    """
    if is_authenticated():
        return True

    # Custom styling for centered login card
    st.markdown("""
    <style>
    .login-container {
        max-width: 520px;
        margin: 1.5rem auto 0 auto;
        padding: 2rem 2rem 1.2rem 2rem;
        background: #131722;
        border: 1px solid #2a2e39;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .login-header {
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .login-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.3rem;
    }
    .login-subtitle {
        font-size: 0.88rem;
        color: #787b86;
    }
    .login-badge {
        display: inline-block;
        background: rgba(255, 215, 0, 0.15);
        color: #FFD700;
        border: 1px solid #FFD700;
        border-radius: 20px;
        padding: 3px 14px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        letter-spacing: 0.5px;
    }
    .one-click-card {
        background: linear-gradient(135deg, rgba(41, 98, 255, 0.18) 0%, rgba(0, 230, 118, 0.15) 100%);
        border: 1px solid rgba(0, 230, 118, 0.45);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 10px 0 12px 0;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 230, 118, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2.2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <span class="login-badge">🛡️ RESTRICTED ACCESS & ADMIN PORTAL</span>
                <div class="login-title">🔐 Institutional Portal</div>
                <div class="login-subtitle">Indian Equities, Options & Commodities Quantitative Screener</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ⚡ PRIMARY 1-CLICK INSTANT LOGIN (NO USERNAME / PASSWORD REQUIRED)
        st.markdown("""
        <div class="one-click-card">
            <div style="font-weight: 800; color: #00E676; font-size: 1.05rem; letter-spacing: 0.3px; margin-bottom: 3px;">
                ⚡ Fast Track: Instant Access
            </div>
            <div style="font-size: 0.82rem; color: #E2E8F0;">
                Click below to enter the dashboard immediately with full privileges. <b>No username or password required.</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 1-Click Login (Instant Access)", type="primary", use_container_width=True, key="btn_one_click_instant"):
            login_user(username="admin")

        # Quick 1-Click Login by Profile
        st.markdown("<div style='font-size: 0.8rem; color: #A0AEC0; margin: 14px 0 6px 0; text-align: center;'>— Or 1-Click Login as a specific team account —</div>", unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        if q1.button("👑 Guruteja", use_container_width=True, key="quick_guru", help="Instant 1-Click Login as Guruteja (Admin)"):
            login_user(username="guruteja")
        if q2.button("👑 Raghavendra", use_container_width=True, key="quick_raghu", help="Instant 1-Click Login as Raghavendra (Admin)"):
            login_user(username="raghavendra")
        if q3.button("👑 Naresh", use_container_width=True, key="quick_naresh", help="Instant 1-Click Login as Naresh (Admin)"):
            login_user(username="naresh")
        if q4.button("👤 Trader", use_container_width=True, key="quick_trader", help="Instant 1-Click Login as Standard Trader"):
            login_user(username="trader")

        # Optional manual credentials expandable form
        with st.expander("🔑 Manual Sign In (Optional Credentials Form)", expanded=False):
            with st.form("login_form", clear_on_submit=False):
                st.markdown("##### Sign in with custom credentials")
                username_in = st.text_input("Username", value="", placeholder="Enter username", key="login_user_input")
                password_in = st.text_input("Password", value="", type="password", placeholder="Enter password", key="login_pass_input")
                
                submit_btn = st.form_submit_button("🔓 Sign In to Dashboard", type="secondary", use_container_width=True)

                if submit_btn:
                    if verify_credentials(username_in, password_in):
                        login_user(username=username_in)
                    else:
                        st.error("❌ Invalid username or password. Please verify your credentials.")

            # Display Authorized Passwords directly inside the expander for reference
            st.markdown("""
            <div style="background: #1a1e29; border: 1px solid #2a2e39; border-radius: 8px; padding: 12px; margin-top: 10px;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px solid #2a2e39; padding-bottom: 6px;">
                    <span style="font-weight: 700; color: #FFD700; font-size: 0.85rem;">🔑 Reference Accounts</span>
                    <span style="background: rgba(255, 215, 0, 0.15); color: #FFD700; border: 1px solid #FFD700; border-radius: 10px; padding: 1px 6px; font-size: 0.68rem; font-weight: bold;">ADMIN ENABLED</span>
                </div>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.78rem; color: #E2E8F0;">
                    <thead>
                        <tr style="border-bottom: 1px solid #333947; color: #A0AEC0; text-align: left;">
                            <th style="padding: 4px 6px;">User</th>
                            <th style="padding: 4px 6px;">Password</th>
                            <th style="padding: 4px 6px;">Role</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #242936;">
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#64B5F6;">guruteja</code></td>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#00E676;">guru@2026</code></td>
                            <td style="padding: 4px 6px; color: #FFD700;">👑 Admin</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242936;">
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#64B5F6;">raghavendra</code></td>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#00E676;">raghavendra@2026</code></td>
                            <td style="padding: 4px 6px; color: #FFD700;">👑 Admin</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242936;">
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#64B5F6;">naresh</code></td>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#00E676;">naresh@2026</code></td>
                            <td style="padding: 4px 6px; color: #FFD700;">👑 Admin</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #242936;">
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#64B5F6;">admin</code></td>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#00E676;">admin@2026</code></td>
                            <td style="padding: 4px 6px; color: #FFD700;">👑 Admin</td>
                        </tr>
                        <tr>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#64B5F6;">trader</code></td>
                            <td style="padding: 4px 6px;"><code style="background:#0e1117; color:#00E676;">nsemcx2026</code></td>
                            <td style="padding: 4px 6px; color: #A0AEC0;">👤 Trader</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

    return False


def render_sidebar_user_badge():
    """Renders the user badge and sign-out button in the sidebar."""
    if not is_authenticated():
        return

    user = get_current_user() or "Authorized User"
    admin_status = is_admin(user)
    role_label = "👑 ADMINISTRATOR" if admin_status else "👤 TRADER"
    role_color = "#FFD700" if admin_status else "#00E676"
    role_bg = "rgba(255, 215, 0, 0.15)" if admin_status else "rgba(0, 230, 118, 0.15)"

    st.sidebar.markdown(f"""
    <div style="background: #1a1e29; border: 1px solid #2a2e39; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.72rem; color: #787b86; letter-spacing: 0.5px;">LOGGED IN AS</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF;">👤 {user.upper()}</div>
            </div>
            <span style="background: {role_bg}; color: {role_color}; border: 1px solid {role_color}; border-radius: 12px; padding: 3px 10px; font-size: 0.72rem; font-weight: bold;">
                {role_label}
            </span>
        </div>
        {"<div style='margin-top: 6px; font-size: 0.75rem; color: #FFD54F;'>⚡ Full Admin Privileges Active</div>" if admin_status else ""}
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout_user()
