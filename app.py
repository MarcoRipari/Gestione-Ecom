import streamlit as st
from streamlit_elements import mui
from streamlit_ace import st_ace
from stqdm import stqdm
from annotated_text import annotated_text
from streamlit_chat import message
import pandas as pd
import plotly.express as px
from zipfile import ZipFile
import io

# ---------------- Configurazione ----------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide", page_icon="👟")

PRIMARY_COLOR = "#4B7BEC"
SECONDARY_COLOR = "#A29BFE"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#2D3436"

# ---------------- Stato Navbar ----------------
if "menu_anchor" not in st.session_state:
    st.session_state.menu_anchor = None
if "selected_menu" not in st.session_state:
    st.session_state.selected_menu = "Home"

def open_menu(event):
    st.session_state.menu_anchor = event.currentTarget

def close_menu():
    st.session_state.menu_anchor = None

def select_menu(item):
    st.session_state.selected_menu = item
    st.session_state.menu_anchor = None

# ---------------- Navbar Material UI ----------------
with mui.AppBar(position="static", sx={"background": PRIMARY_COLOR}):
    with mui.Toolbar():
        mui.Typography("Generatore Descrizioni 👟", variant="h6", sx={"flexGrow": 1})

        mui.Button("Home", color="inherit", onClick=lambda e: select_menu("Home"))
        mui.Button("Upload", color="inherit", onClick=lambda e: select_menu("Upload"))

        # Dropdown Generazione
        mui.Button("Generazione ⬇️", color="inherit", onClick=open_menu)
        with mui.Menu(
            anchorEl=st.session_state.menu_anchor,
            open=bool(st.session_state.menu_anchor),
            onClose=close_menu
        ):
            mui.MenuItem("Descrizioni", onClick=lambda e: select_menu("Generazione - Descrizioni"))
            mui.MenuItem("Nomi", onClick=lambda e: select_menu("Generazione - Nomi"))
            mui.MenuItem("Cognomi", onClick=lambda e: select_menu("Generazione - Cognomi"))

        mui.Button("Storico", color="inherit", onClick=lambda e: select_menu("Storico"))
        mui.Button("Download", color="inherit", onClick=lambda e: select_menu("Download"))
        mui.Button("Chat AI", color="inherit", onClick=lambda e: select_menu("Chat AI"))

# ---------------- Contenuto dinamico ----------------
if st.session_state.selected_menu == "Home":
    st.markdown(f"""
    <div style="background-color:{CARD_BG};padding:50px;border-radius:15px;
                box-shadow:0 6px 12px rgba(0,0,0,0.1);text-align:center;">
        <h1 style="color:{PRIMARY_COLOR};margin-bottom:10px;">Generatore Descrizioni Calzature</h1>
        <p style="color:{TEXT_COLOR};font-size:18px;">Benvenuto! Usa il menu in alto per navigare.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.selected_menu == "Upload":
    st.header("📂 Carica il tuo CSV")
    uploaded_file = st.file_uploader("Seleziona il file CSV con la colonna SKU", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File caricato con successo!")
        st.dataframe(df, use_container_width=True)

elif st.session_state.selected_menu == "Generazione - Descrizioni":
    st.header("⚙️ Generazione Descrizioni")
    prompt = st_ace(value="Scrivi qui il prompt per le descrizioni...", language="markdown", theme="chrome", height=200)
    with st.form("desc_form"):
        langs = st.multiselect("Lingue disponibili", ["it", "en", "fr", "de"], default=["it"])
        batch_size = st.slider("Batch size", 1, 50, 5)
        submit = st.form_submit_button("🚀 Avvia")
    if submit:
        for _ in stqdm(range(batch_size), desc="Generazione descrizioni"):
            pass
        st.success("✅ Descrizioni generate!")

elif st.session_state.selected_menu == "Generazione - Nomi":
    st.header("⚙️ Generazione Nomi")
    st.write("Funzione di generazione nomi (logica invariata da implementare qui).")

elif st.session_state.selected_menu == "Generazione - Cognomi":
    st.header("⚙️ Generazione Cognomi")
    st.write("Funzione di generazione cognomi (logica invariata da implementare qui).")

elif st.session_state.selected_menu == "Storico":
    st.header("📊 Storico & RAG")
    demo_df = pd.DataFrame({"Lingua": ["it", "en", "fr"], "Descrizioni": [120, 95, 60]})
    fig = plotly.express.bar(demo_df, x="Lingua", y="Descrizioni", title="Distribuzione descrizioni per lingua")
    st.plotly_chart(fig, use_container_width=True)
    annotated_text(("SKU123", "Scarpa elegante", "#faa"), " generata con successo.")

elif st.session_state.selected_menu == "Download":
    st.header("⬇️ Download CSV / ZIP")
    with io.BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            z.writestr("example.csv", "SKU,Descrizione\n123,Scarpa esempio")
        buffer.seek(0)
        st.download_button("📦 Scarica ZIP", buffer, file_name="descrizioni.zip")

elif st.session_state.selected_menu == "Chat AI":
    st.header("💬 Chat con AI")
    message("Ciao! Vuoi un feedback sulle tue descrizioni?", is_user=False)
    user_input = st.text_input("Scrivi qui")
    if user_input:
        message(user_input, is_user=True)
        message("Risposta AI generica...", is_user=False)
