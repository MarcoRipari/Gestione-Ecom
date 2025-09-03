"""
Streamlit Pro UI/UX Template – Refined Professional Edition
===========================================================

✨ Polished design for business use
✨ Modern, clean topbar with branding
✨ Tailwind injection fixed (reliable CDN)
✨ Sidebar with professional card layout
✨ Responsive, corporate-friendly aesthetic

"""

from __future__ import annotations
import time
from datetime import datetime
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Imports with safe fallbacks
# -----------------------------
try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

# (Other optional imports skipped for brevity; same as before)

# -------------------------------------------------
# Global App Config
# -------------------------------------------------
st.set_page_config(
    page_title="Pro Business App",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Tailwind Injection (fixed)
# -------------------------------------------------
TAILWIND_CDN = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"

st.markdown(
    f"""
    <link rel="stylesheet" href="{TAILWIND_CDN}">
    <style>
        body {{ background-color: #f9fafb; }}
        .topbar {{
            background: linear-gradient(90deg, #111827, #1f2937);
            color: #f9fafb;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.8rem 1.5rem;
            border-bottom: 1px solid #2d3748;
        }}
        .topbar-title {{
            font-size: 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }}
        .brand-logo {{
            width: 28px;
            height: 28px;
        }}
        .status-chip {{
            background: #10b981;
            color: white;
            font-size: 0.75rem;
            padding: 2px 10px;
            border-radius: 999px;
        }}
        .sidebar-card {{
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Demo Data
# -------------------------------------------------
np.random.seed(42)
demo_df = pd.DataFrame({
    "SKU": [f"SKU-{i:04d}" for i in range(1, 51)],
    "Brand": np.random.choice(["Aurora", "Nimbus", "Solace"], 50),
    "Price": np.random.uniform(39, 199, 50).round(2),
    "Stock": np.random.randint(0, 500, 50),
    "Updated": pd.Timestamp("2025-08-25") + pd.to_timedelta(np.random.randint(0, 7, 50), unit="D"),
})

# -------------------------------------------------
# Session User Model
# -------------------------------------------------
@dataclass
class User:
    name: str
    role: str = "viewer"
    avatar_url: str = "https://cdn-icons-png.flaticon.com/512/149/149071.png"


def get_user() -> User | None:
    return st.session_state.get("user")


def set_user(name: str, role: str = "viewer"):
    st.session_state["user"] = User(name=name, role=role)

# -------------------------------------------------
# Header / Topbar
# -------------------------------------------------
def topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-title">
                <img src="https://cdn-icons-png.flaticon.com/512/3652/3652191.png" alt="logo" class="brand-logo" />
                <span>Pro Business App</span>
                <span class="status-chip">beta</span>
            </div>
            <div>
                <input type="text" placeholder="Search…" style="padding:6px 10px;border-radius:8px;border:1px solid #374151;background:#1f2937;color:#f9fafb;" />
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------
# Sidebar (Navigation + Status)
# -------------------------------------------------
SIDEBAR_PAGES = ["Dashboard", "Data", "Editor", "Chat", "Media", "Settings", "About"]


def sidebar_menu() -> str:
    with st.sidebar:
        st.markdown("<div class='sidebar-card'><h3 style='margin:0;'>Navigation</h3></div>", unsafe_allow_html=True)
        if option_menu:
            selected = option_menu(
                menu_title="",
                options=SIDEBAR_PAGES,
                icons=["speedometer2", "table", "code", "chat-dots", "camera-video", "gear", "info-circle"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important"},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin":"2px 0"},
                    "icon": {"font-size": "16px"},
                    "nav-link-selected": {"background-color": "#2563eb", "color": "white"},
                },
            )
        else:
            selected = st.selectbox("Navigation", SIDEBAR_PAGES)

        st.markdown("<div class='sidebar-card'><h4>Status</h4>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Users", "128", "+3")
        c2.metric("Jobs", "512", "+12")
        st.markdown("</div>", unsafe_allow_html=True)

        return selected

# -------------------------------------------------
# Pages (placeholders)
# -------------------------------------------------
def page_dashboard():
    st.subheader("Dashboard")
    st.info("This is a professional dashboard placeholder.")

def page_data():
    st.subheader("Data")
    st.dataframe(demo_df, use_container_width=True)

def page_editor():
    st.subheader("Editor")
    st.text_area("Editor", "# Write something...", height=200)

def page_chat():
    st.subheader("Chat")
    st.text_input("Type a message")

def page_media():
    st.subheader("Media")
    st.warning("WebRTC disabled in demo.")

def page_settings():
    st.subheader("Settings")
    st.text_input("Workspace name")

def page_about():
    st.subheader("About")
    st.write("Professional App Template – clean, modern and extensible.")

PAGE_MAP = {
    "Dashboard": page_dashboard,
    "Data": page_data,
    "Editor": page_editor,
    "Chat": page_chat,
    "Media": page_media,
    "Settings": page_settings,
    "About": page_about,
}

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    topbar()
    page = sidebar_menu()

    with st.container():
        PAGE_MAP.get(page, page_dashboard)()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"© {datetime.now().year} Pro Business App. Built with Streamlit.")

if __name__ == "__main__":
    main()
