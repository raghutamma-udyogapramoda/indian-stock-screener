"""
Authentication and Access Control Manager for Streamlit Application.
Provides institutional-grade user authentication, password verification,
and session management via Streamlit Secrets and environment variables.
"""

import hashlib
import os
import time
from typing import Dict, Optional
import streamlit as st


# Default fallback credentials for local testing if no secrets or env vars are configured
DEFAULT_USERS = {
    "admin": "admin@2026",
    "trader": "nsemcx2026",
    "raghavendra": "screener2026"
}


def _hash_password(password: str) -> str:
    """Computes SHA-256 hash of a password."""
    return hashlib.sha256(password.strip().encode("utf-8")).hexdigest()


def get_user_database() -> Dict[str, str]:
    """
    Loads authorized users from:
    1. Streamlit Secrets (st.secrets["users"])
    2. Environment variables (ADMIN_USERNAME / ADMIN_PASSWORD or APP_USERS)
    3. Default built-in accounts (for local initial testing)
    """
    users: Dict[str, str] = {}

    # 1. Check Streamlit Secrets
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

    # 3. If no users configured anywhere, load default users
    if not users:
        users = {u.lower(): p for u, p in DEFAULT_USERS.items()}

    return users


def verify_credentials(username: str, password_attempt: str) -> bool:
    """
    Verifies username and password against configured users.
    Supports both plain text and SHA-256 hashed passwords in secrets.
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


def logout_user():
    """Logs out the current user and clears session state."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state.pop("cached_results", None)
    st.session_state.pop("symbols", None)
    st.session_state.pop("data_dict", None)
    st.rerun()


def render_login_gate() -> bool:
    """
    Renders the login page if the user is not authenticated.
    Returns True if user is authenticated, False otherwise (stops script execution).
    """
    if is_authenticated():
        return True

    # Custom styling for centered login card
    st.markdown("""
    <style>
    .login-container {
        max-width: 480px;
        margin: 2rem auto;
        padding: 2.2rem;
        background: #131722;
        border: 1px solid #2a2e39;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .login-header {
        text-align: center;
        margin-bottom: 1.5rem;
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
        background: rgba(41, 98, 255, 0.15);
        color: #2962FF;
        border: 1px solid #2962FF;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <span class="login-badge">🛡️ RESTRICTED ACCESS</span>
                <div class="login-title">🔐 Institutional Portal</div>
                <div class="login-subtitle">Indian Equities & Derivatives Quantitative Screener</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### Sign in with authorized credentials")
            username_in = st.text_input("Username", placeholder="Enter username", key="login_user_field")
            password_in = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass_field")
            
            submit_btn = st.form_submit_button("🔓 Sign In to Dashboard", type="primary", use_container_width=True)

            if submit_btn:
                if verify_credentials(username_in, password_in):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username_in.strip()
                    st.session_state["login_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.toast(f"✅ Welcome, {username_in}! Loading screener...", icon="🚀")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password. Please verify your credentials.")

        st.caption("🔒 Access restricted to authorized traders. Configured via Streamlit Secrets or Environment.")

        # If running with default built-in credentials, display helpful helper banner for developer
        db = get_user_database()
        if "admin" in db and db["admin"] == DEFAULT_USERS["admin"]:
            with st.expander("ℹ️ Initial Setup / Default Access Credentials"):
                st.markdown("""
                **Default Credentials (for initial setup & testing):**
                - `admin` / `admin@2026`
                - `raghavendra` / `screener2026`
                - `trader` / `nsemcx2026`

                *To configure custom private accounts, add `[users]` in `.streamlit/secrets.toml` or Streamlit Cloud Secrets settings.*
                """)

    return False


def render_sidebar_user_badge():
    """Renders the user badge and sign-out button in the sidebar."""
    if not is_authenticated():
        return

    user = get_current_user() or "Authorized User"
    st.sidebar.markdown(f"""
    <div style="background: #1a1e29; border: 1px solid #2a2e39; border-radius: 8px; padding: 10px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.75rem; color: #787b86;">LOGGED IN AS</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #00E676;">👤 {user.upper()}</div>
            </div>
            <span style="background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid #00E676; border-radius: 12px; padding: 2px 8px; font-size: 0.7rem; font-weight: bold;">
                ACTIVE
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout_user()
