from __future__ import annotations
import streamlit as st
from dataclasses import dataclass
import pandas as pd
import numpy as np

try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

st.set_page_config(page_title="Pro Business App", page_icon="💼", layout="wide")

# Demo data
np.random.seed(42)
demo_df = pd.DataFrame({"SKU": [f"SKU-{i:04d}" for i in range(1, 51)],
                        "Brand": np.random.choice(["Aurora", "Nimbus", "Solace"], 50),
                        "Price": np.random.uniform(39,199,50).round(2),
                        "Stock": np.random.randint(0,500,50)})

@dataclass
class User:
    name: str
    role: str = "viewer"

# Topbar

def topbar():
    st.markdown("""
        <div style='display:flex;justify-content:space-between;align-items:center;padding:0.8rem 1.5rem;background:#1f2937;color:white;'>
            <div style='display:flex;align-items:center;gap:0.6rem;font-weight:600;'>
                <img src='https://cdn-icons-png.flaticon.com/512/3652/3652191.png' width='28' height='28'/>
                <span>Pro Business App</span>
            </div>
            <input type='text' placeholder='Search…' style='padding:6px 10px;border-radius:8px;border:none;background:#374151;color:white;' />
        </div>
    """, unsafe_allow_html=True)

# Sidebar with submenu support

def sidebar_menu():
    with st.sidebar:
        st.write("### Navigation")
        main_menu = option_menu(
            menu_title=None,
            options=["Dashboard", "Data", "Editor", "Chat", "Media", "Settings", "About"],
            icons=["speedometer2", "table", "code", "chat-dots", "camera-video", "gear", "info-circle"],
            default_index=0,
            styles={"nav-link-selected": {"background-color": "#2563eb", "color": "white"}}
        )

        # Example of submenus
        if main_menu == "Data":
            submenu = option_menu(
                menu_title="Data Options",
                options=["Overview", "Analytics", "Export"],
                icons=["eye", "bar-chart", "download"],
                default_index=0,
                orientation="vertical",
                styles={"nav-link-selected": {"background-color": "#10b981", "color": "white"}}
            )
            return main_menu, submenu
        return main_menu, None

# Pages placeholders

def page_dashboard(): st.subheader("Dashboard")
def page_data(submenu=None): st.subheader(f"Data - {submenu or 'Overview'}")
def page_editor(): st.subheader("Editor")
def page_chat(): st.subheader("Chat")
def page_media(): st.subheader("Media")
def page_settings(): st.subheader("Settings")
def page_about(): st.subheader("About")

PAGE_MAP = {
    "Dashboard": page_dashboard,
    "Data": page_data,
    "Editor": page_editor,
    "Chat": page_chat,
    "Media": page_media,
    "Settings": page_settings,
    "About": page_about,
}

# Main

def main():
    topbar()
    page, submenu = sidebar_menu()
    if page == "Data":
        PAGE_MAP[page](submenu)
    else:
        PAGE_MAP[page]()

if __name__ == "__main__":
    main()
