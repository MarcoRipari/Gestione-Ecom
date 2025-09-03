"""
Streamlit Pro UI/UX Template – Multi-User, Component-Rich
=========================================================

✅ Ready for Streamlit Cloud
✅ Modular pages inside one file (you can split later)
✅ Showcases these components:
   - streamlit-option-menu
   - streamlit-aggrid
   - streamlit-ace
   - st-tailwind
   - streamlit-elements
   - streamlit-modal
   - stqdm
   - plotly / plotly-express
   - altair
   - streamlit-echarts
   - st-annotated-text
   - streamlit-chat
   - streamlit-webrtc

Dependencies (requirements.txt example):
---------------------------------------
streamlit>=1.37
pandas
numpy
plotly
altair
streamlit-option-menu
streamlit-aggrid
streamlit-ace
streamlit-elements
streamlit-modal
stqdm
streamlit-echarts
annotated-text
streamlit-chat
streamlit-webrtc
# Optional: st-tailwind (or serve Tailwind CDN below)

Notes:
------
- This template uses graceful fallbacks if an optional component is missing.
- Replace placeholder assets (logo/avatar URLs) with your own.
- Authentication here is demo-level (session based). Integrate your SSO/DB later.
"""

from __future__ import annotations
import os
import time
from datetime import datetime
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Optional/3rd‑party components
# -----------------------------
# Each import is wrapped with try/except to allow the app to run even if a package is missing.
try:
    from streamlit_option_menu import option_menu
except Exception:  # pragma: no cover
    option_menu = None

try:
    from st_aggrid import AgGrid, GridOptionsBuilder
except Exception:  # pragma: no cover
    AgGrid, GridOptionsBuilder = None, None

try:
    from streamlit_ace import st_ace
except Exception:  # pragma: no cover
    st_ace = None

try:
    # Package often named "st-tailwind"; import path may vary by release.
    # If unavailable, we'll use a CDN injection fallback.
    from st_tailwind import st_tailwind
except Exception:  # pragma: no cover
    st_tailwind = None

try:
    from streamlit_elements import elements, mui, html, dashboard
except Exception:  # pragma: no cover
    elements = mui = html = dashboard = None

try:
    from streamlit_modal import Modal
except Exception:  # pragma: no cover
    Modal = None

try:
    from stqdm import stqdm
except Exception:  # pragma: no cover
    stqdm = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = go = None

try:
    import altair as alt
except Exception:  # pragma: no cover
    alt = None

try:
    from streamlit_echarts import st_echarts
except Exception:  # pragma: no cover
    st_echarts = None

try:
    from annotated_text import annotated_text
except Exception:  # pragma: no cover
    annotated_text = None

try:
    from streamlit_chat import message as chat_message
except Exception:  # pragma: no cover
    chat_message = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
except Exception:  # pragma: no cover
    webrtc_streamer = WebRtcMode = None

# -------------------------------------------------
# Global App Config (Streamlit Cloud friendly)
# -------------------------------------------------
st.set_page_config(
    page_title="Pro App Template",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------
# Tailwind & Theme utilities
# -------------------------------------------------
TAILWIND_CDN = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"

def inject_tailwind():
    """Inject Tailwind CSS via component (preferred) or CDN fallback."""
    if st_tailwind:
        try:
            st_tailwind()  # provides Tailwind utilities inside Streamlit
        except Exception:
            _tailwind_cdn_fallback()
    else:
        _tailwind_cdn_fallback()


def _tailwind_cdn_fallback():
    st.markdown(
        f"""
        <link rel="stylesheet" href="{TAILWIND_CDN}">
        <style>
            /* Additional design tokens */
            :root {{
                --brand: #6D28D9; /* violet-700 */
                --brand-600: #7C3AED; /* violet-600 */
                --muted: #64748B;    /* slate-500 */
                --bg: #0B1220;       /* dark surface */
            }}
            .glass {{ backdrop-filter: blur(10px); background: rgba(255,255,255,0.08); }}
            .chip {{ display:inline-block; padding:4px 10px; border-radius:999px; background:#1f2937; color:#e5e7eb; font-size:12px; }}
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
# Minimal session user model
# -------------------------------------------------
@dataclass
class User:
    name: str
    role: str = "viewer"
    avatar_url: str = "https://avatars.githubusercontent.com/u/9919?s=200&v=4"  # GH logo as placeholder


def get_user() -> User | None:
    return st.session_state.get("user")


def set_user(name: str, role: str = "viewer"):
    st.session_state["user"] = User(name=name, role=role)


# -------------------------------------------------
# Header / Topbar
# -------------------------------------------------
def topbar():
    with st.container():
        st.markdown(
            """
            <div class="w-full flex items-center justify-between py-3 px-4 md:px-6 border-b border-slate-800" style="background:linear-gradient(90deg,#0b1020,#111827);">
              <div class="flex items-center gap-3">
                <img src="https://em-content.zobj.net/source/microsoft-teams/363/jigsaw-puzzle-piece_1f9e9.png" alt="logo" width="28" height="28"/>
                <div class="text-slate-50 text-lg md:text-xl font-semibold">Pro App Template</div>
                <span class="chip ml-2">beta</span>
              </div>
              <div class="flex items-center gap-3">
                <input id="search" placeholder="Search…" class="hidden md:block rounded-xl px-3 py-1 bg-slate-900 text-slate-200 border border-slate-700"/>
                <a class="chip" href="https://docs.streamlit.io" target="_blank">Docs</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------------------------------------
# Sidebar (Navigation + Status)
# -------------------------------------------------
SIDEBAR_PAGES = [
    "Dashboard",
    "Data",
    "Editor",
    "Chat",
    "Media",
    "Elements",
    "Settings",
    "About",
]


def sidebar_menu() -> str:
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1520975916090-3105956dac38?w=640&q=80", caption="", use_column_width=True)
        st.markdown("<div class='mt-2'></div>", unsafe_allow_html=True)

        if option_menu:
            selected = option_menu(
                menu_title="Navigation",
                options=SIDEBAR_PAGES,
                icons=["speedometer2", "table", "code", "chat-dots", "camera-video", "grid", "gear", "info-circle"],
                default_index=0,
                styles={
                    "container": {"padding": "0!important"},
                    "nav-link": {"font-size": "14px", "text-align": "left"},
                    "icon": {"font-size": "16px"},
                    "nav-link-selected": {"background-color": "#1f2937"},
                },
            )
        else:
            selected = st.selectbox("Navigation", SIDEBAR_PAGES)

        # User card
        user = get_user()
        if user:
            st.markdown(f"**User:** {user.name} · *{user.role}*")
            st.image(user.avatar_url, width=48)
            if st.button("Sign out"):
                st.session_state.pop("user", None)
                st.rerun()
        else:
            with st.form("login_form", clear_on_submit=True):
                st.subheader("Sign in")
                name = st.text_input("Name", "")
                role = st.selectbox("Role", ["viewer", "editor", "admin"])
                submitted = st.form_submit_button("Sign in")
                if submitted and name:
                    set_user(name, role)
                    st.success(f"Welcome, {name}!")
                    st.rerun()

        st.divider()
        st.caption("Status")
        kpi1, kpi2 = st.columns(2)
        kpi1.metric("Users", "128", "+3")
        kpi2.metric("Jobs", "512", "+12")
        return selected


# -------------------------------------------------
# Dashboard Page
# -------------------------------------------------

def page_dashboard():
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Active Projects", 24, 2)
    with c2:
        st.metric("Success Rate", "98.2%", "+0.3%")
    with c3:
        st.metric("Avg. Latency", "423 ms", "-12 ms")

    # Plotly chart
    if px:
        st.markdown("### Plotly – Performance")
        df = pd.DataFrame({
            "date": pd.date_range(datetime.now().date(), periods=14),
            "requests": np.random.randint(200, 600, 14),
        })
        fig = px.line(df, x="date", y="requests")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Altair chart
    if alt:
        st.markdown("### Altair – Stock by Brand")
        ch = (
            alt.Chart(demo_df)
            .mark_bar()
            .encode(x="Brand:N", y="sum(Stock):Q", tooltip=["Brand", "sum(Stock)"])
            .properties(height=300)
        )
        st.altair_chart(ch, use_container_width=True)

    # ECharts example
    if st_echarts:
        st.markdown("### ECharts – Donut")
        option = {
            "tooltip": {"trigger": "item"},
            "series": [
                {
                    "name": "Brand",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "label": {"show": False, "position": "center"},
                    "emphasis": {"label": {"show": True, "fontSize": 18, "fontWeight": "bold"}},
                    "labelLine": {"show": False},
                    "data": [
                        {"value": int(demo_df[demo_df.Brand=="Aurora"].shape[0]), "name": "Aurora"},
                        {"value": int(demo_df[demo_df.Brand=="Nimbus"].shape[0]), "name": "Nimbus"},
                        {"value": int(demo_df[demo_df.Brand=="Solace"].shape[0]), "name": "Solace"},
                    ],
                }
            ],
        }
        st_echarts(option, height="300px")


# -------------------------------------------------
# Data Page (AgGrid)
# -------------------------------------------------

def page_data():
    st.subheader("Data Table")
    if AgGrid and GridOptionsBuilder:
        gob = GridOptionsBuilder.from_dataframe(demo_df)
        gob.configure_pagination(enabled=True)
        gob.configure_default_column(resizable=True, filter=True, sortable=True)
        gob.configure_selection("multiple", use_checkbox=True)
        grid_options = gob.build()
        grid_response = AgGrid(
            demo_df,
            gridOptions=grid_options,
            height=420,
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            allow_unsafe_jscode=True,
        )
        sel = grid_response.get("selected_rows", [])
        st.info(f"Selected rows: {len(sel)}")
    else:
        st.warning("streamlit-aggrid not installed – showing fallback table.")
        st.dataframe(demo_df, use_container_width=True)

    st.markdown("### Annotated Text")
    if annotated_text:
        annotated_text(
            ("Aurora X1", "SKU", "violet"),
            " has ",
            ("low stock", "status", "red"),
            ", consider reorder by ",
            ("Friday", "deadline", "#6EE7B7"),
            ".",
        )
    else:
        st.write("*Install `annotated-text` to see annotations here.*")


# -------------------------------------------------
# Editor Page (ACE + stqdm + Modal)
# -------------------------------------------------

def page_editor():
    st.subheader("Content Editor")

    if st_ace:
        code = st_ace(
            value="""# Write your prompt or code here\nprint('Hello, Streamlit ACE!')\n""",
            language="python",
            theme="twilight",
            keybinding="vscode",
            min_lines=8,
            max_lines=24,
            wrap=True,
        )
        st.caption("Ctrl/Cmd+Enter to run (demo)")
        if st.button("Run Snippet"):
            st.code(code or "# (empty)")
    else:
        st.warning("streamlit-ace not installed – showing fallback text area.")
        st.text_area("Editor", "print('Hello!')", height=200)

    st.divider()
    st.subheader("Batch Demo with stqdm")
    n = st.slider("Items", 5, 50, 10)
    logs = []
    if st.button("Process"):
        if stqdm:
            for i in stqdm(range(n), desc="Processing"):
                time.sleep(0.05)
                logs.append(f"Item {i} done")
        else:
            prog = st.progress(0)
            for i in range(n):
                time.sleep(0.05)
                prog.progress(int((i+1)/n*100))
                logs.append(f"Item {i} done")
        st.success("Completed!")
        st.code("\n".join(logs[-10:]), language="text")

    if Modal:
        if st.button("Open Modal"):
            modal = Modal("Quick Action", key="modal-key", max_width=600)
            with modal.container():
                st.markdown("### This is a modal dialog")
                st.write("Use it for confirmations, previews, etc.")
                st.button("Close")
    else:
        st.info("Install `streamlit-modal` to use modals.")


# -------------------------------------------------
# Chat Page (streamlit-chat)
# -------------------------------------------------

def page_chat():
    st.subheader("Team Chat")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Ciao! Come posso aiutarti oggi?"}
        ]

    if chat_message:
        for i, msg in enumerate(st.session_state.chat_history):
            chat_message(msg["content"], is_user=(msg["role"] == "user"), key=f"chat-{i}")
        user_in = st.text_input("Type a message", key="chat-input")
        if st.button("Send", type="primary") and user_in:
            st.session_state.chat_history.append({"role": "user", "content": user_in})
            st.session_state.chat_history.append({"role": "assistant", "content": "(demo) Ricevuto!"})
            st.rerun()
    else:
        st.warning("Install `streamlit-chat` to enable chat bubbles.")
        st.text_input("Type a message")


# -------------------------------------------------
# Media Page (WebRTC)
# -------------------------------------------------

def page_media():
    st.subheader("Media & RTC")
    if webrtc_streamer and WebRtcMode:
        st.info("Camera demo – your browser may prompt for permission.")
        webrtc_streamer(key="webrtc", mode=WebRtcMode.SENDRECV, media_stream_constraints={"video": True, "audio": False})
    else:
        st.warning("Install `streamlit-webrtc` to enable camera/mic.")


# -------------------------------------------------
# Elements Page (streamlit-elements: resizable dashboard)
# -------------------------------------------------

def page_elements():
    st.subheader("Resizable Dashboard – streamlit-elements")
    if not elements:
        st.warning("Install `streamlit-elements` to use this section.")
        return

    # A simple draggable/resizable layout with one chart and one HTML card
    with elements("demo"):
        layout = [
            dashboard.Item("chart", 0, 0, 6, 6),
            dashboard.Item("card", 6, 0, 6, 6),
        ]
        with dashboard.Grid(layout, draggableHandle=".draggable"):
            with mui.Card(key="chart", className="draggable"):
                with mui.CardContent():
                    st.write("Plotly Bar (inside elements)")
                    if px:
                        fig = px.bar(demo_df.groupby("Brand").Stock.sum().reset_index(), x="Brand", y="Stock")
                        st.plotly_chart(fig, use_container_width=True)
            with mui.Card(key="card", className="draggable"):
                with mui.CardHeader(title="Notes", subheader=str(datetime.now().date())):
                    pass
                with mui.CardContent():
                    st.write("Drag me around. Resize from corners.")
                    html.div("""
                        <div style='padding:8px;border-radius:12px;background:#111827;color:#e5e7eb'>
                            <b>Tip:</b> Use <code>streamlit-elements</code> for complex UIs.
                        </div>
                    """)


# -------------------------------------------------
# Settings Page
# -------------------------------------------------

def page_settings():
    st.subheader("Settings")
    theme = st.radio("Theme", ["Auto", "Light", "Dark"], horizontal=True)
    st.toggle("Compact mode", value=False)
    st.text_input("Workspace name", value="My Team")
    st.text_input("API key (placeholder)", type="password")
    st.caption("These are visual/demo settings only.")


# -------------------------------------------------
# About Page
# -------------------------------------------------

def page_about():
    st.subheader("About")
    st.markdown(
        """
        **Pro App Template** – a production‑minded UI starter for multi‑user Streamlit apps.

        - Clean layout with topbar + sidebar
        - Rich component gallery with graceful fallbacks
        - Easy to split into `/pages` modules later
        - Ready for Streamlit Cloud
        """
    )


# -------------------------------------------------
# Router
# -------------------------------------------------
PAGE_MAP = {
    "Dashboard": page_dashboard,
    "Data": page_data,
    "Editor": page_editor,
    "Chat": page_chat,
    "Media": page_media,
    "Elements": page_elements,
    "Settings": page_settings,
    "About": page_about,
}


def router(page_name: str):
    fn = PAGE_MAP.get(page_name, page_dashboard)
    fn()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    inject_tailwind()
    topbar()
    page = sidebar_menu()

    # Gate pages by pseudo-role (demo)
    user = get_user()
    if page in {"Editor", "Elements", "Settings"} and (not user or user.role not in {"editor", "admin"}):
        st.error("You need editor or admin access to view this page.")
        return

    with st.container():
        router(page)

    st.markdown("<div class='py-8'></div>", unsafe_allow_html=True)
    st.caption("© {} Pro App Template. Built with Streamlit.".format(datetime.now().year))


if __name__ == "__main__":
    main()
