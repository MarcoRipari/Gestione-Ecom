import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
from stqdm import stqdm
from st_annotated_text import annotated_text
from streamlit_chat import message
import pandas as pd
import plotly.express as px
from zipfile import ZipFile
import io

# ---------------- Tema personalizzabile ----------------
PRIMARY_COLOR = "#4B7BEC"
SECONDARY_COLOR = "#A29BFE"
CARD_BG = "#FFFFFF"
BACKGROUND_COLOR = "#F0F2F6"
TEXT_COLOR = "#2D3436"

st.set_page_config(page_title="Generatore Descrizioni", layout="wide", page_icon="👟")

# ---------------- Header ----------------
st.markdown(f"""
<div style="background-color:{PRIMARY_COLOR};padding:25px;border-radius:15px;text-align:center;">
<h1 style="color:white;margin:0;">Generatore Descrizioni Calzature</h1>
<p style="color:white;margin:0;">Elegante, moderno e user-friendly</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar Menu ----------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Upload CSV", "Generazione", "Storico / RAG", "Download", "Chat AI"],
        icons=["cloud-upload", "gear", "folder", "download", "chat"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": BACKGROUND_COLOR},
            "icon": {"color": PRIMARY_COLOR, "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "--hover-color": SECONDARY_COLOR},
            "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
        }
    )

st.markdown(f"<div style='background-color:{BACKGROUND_COLOR};padding:20px;border-radius:15px'>", unsafe_allow_html=True)

# ---------------- Upload CSV ----------------
if selected == "Upload CSV":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:20px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.subheader("Carica il tuo CSV")
    uploaded_file = st.file_uploader("Seleziona il file CSV con la colonna SKU", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Anteprima CSV")
        st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Generazione ----------------
elif selected == "Generazione":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:20px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.subheader("Prompt & Configurazione Generazione")
    
    st.markdown("### Modifica Prompt")
    prompt = st_ace(value="Prompt di esempio...", language="python", theme="chrome", height=150)
    
    with st.form("gen_form"):
        langs = st.multiselect("Lingue", ["it", "en", "fr", "de"], default=["it"])
        batch_size = st.number_input("Batch size", min_value=1, value=5)
        submit_gen = st.form_submit_button("Genera Descrizioni")
    
    if submit_gen:
        st.info("Generazione in corso...")
        for i in stqdm(range(batch_size), desc="Processing"):
            pass  # Qui chiami la tua funzione di generazione originale
        st.success("Generazione completata!")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Storico / RAG ----------------
elif selected == "Storico / RAG":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:20px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.subheader("Storico / RAG")
    st.write("Visualizza storico interattivo")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Download ----------------
elif selected == "Download":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:20px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.subheader("Download CSV / ZIP")
    with io.BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            z.writestr("example.csv", "SKU,Descrizione\n123,Scarpa esempio")
        buffer.seek(0)
        st.download_button("Scarica ZIP", buffer, file_name="descrizioni.zip", key="download_zip")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Chat AI ----------------
elif selected == "Chat AI":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:20px;border-radius:15px;box-shadow:0 4px 6px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.subheader("Chat con AI")
    message("Ciao! Chiedimi qualcosa sulle descrizioni generate.", is_user=False)
    user_input = st.text_input("Scrivi qui")
    if user_input:
        message(user_input, is_user=True)
        message("Risposta AI generica...", is_user=False)
    st.markdown("</div>", unsafe_allow_html=True)
