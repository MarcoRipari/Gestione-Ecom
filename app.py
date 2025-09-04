import streamlit as st
from streamlit_option_menu import option_menu

# =========================
# CONFIGURAZIONE BASE
# =========================
st.set_page_config(
    page_title="Demo App",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STILI PERSONALIZZATI
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        /* Font & background */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #f7f9fc;
            color: #1a1a1a;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e5e5;
        }

        /* Card style */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Buttons */
        div.stButton > button {
            border-radius: 10px;
            padding: 0.6em 1.2em;
            background-color: #4a90e2;
            color: white;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #357ab8;
            color: white;
        }

        /* Footer */
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #888;
            padding: 1rem 0;
            border-top: 1px solid #e5e5e5;
            margin-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css()

# =========================
# SIDEBAR MENU
# =========================
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=120)
    selected = option_menu(
        menu_title="Navigazione",
        options=["Home", "Dashboard", "Impostazioni"],
        icons=["house", "bar-chart", "gear"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    st.markdown("### Info")
    st.markdown("Applicativo demo con UI/UX moderna.")

# =========================
# HEADER
# =========================
st.title("✨ Applicativo Demo")
st.markdown("Interfaccia moderna, elegante e professionale, pronta all'uso.")

# =========================
# MAIN CONTENT
# =========================
if selected == "Home":
    st.header("🏠 Home")
    st.write("Benvenuto nella tua nuova applicazione Streamlit con layout personalizzato.")
    st.button("Inizia subito")

elif selected == "Dashboard":
    st.header("📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Utenti", "1,024", "+12%")
    with col2:
        st.metric("Entrate", "€ 8,560", "+8%")
    with col3:
        st.metric("Conversioni", "4.3%", "-1%")

    st.line_chart({"Vendite": [3, 6, 9, 12, 8, 5, 7]})

elif selected == "Impostazioni":
    st.header("⚙️ Impostazioni")
    st.text_input("Nome applicazione", value="Demo App")
    st.color_picker("Colore tema", "#4a90e2")

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class="footer">
        Creato con ❤️ usando Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
