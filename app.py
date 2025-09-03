import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
from stqdm import stqdm
from streamlit_chat import message
import pandas as pd
import plotly.express as px
from zipfile import ZipFile
import io

# ---------------- Tema personalizzabile ----------------
PRIMARY_COLOR = "#4B7BEC"
SECONDARY_COLOR = "#A29BFE"
BACKGROUND_COLOR = "#F7F9FC"
TEXT_COLOR = "#2D3436"

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")

# ---------------- Header ----------------
st.markdown(f"""
<div style="background-color:{PRIMARY_COLOR};padding:20px;border-radius:10px">
<h1 style="color:white;text-align:center;">Generatore Descrizioni Calzature</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar Menu ----------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Upload CSV", "Generazione", "Storico / RAG", "Download"],
        icons=["cloud-upload", "gear", "folder", "download"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": BACKGROUND_COLOR},
            "icon": {"color": PRIMARY_COLOR, "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "--hover-color": SECONDARY_COLOR},
            "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
        }
    )

# ---------------- Upload CSV ----------------
if selected == "Upload CSV":
    st.subheader("Carica il tuo CSV")
    uploaded_file = st.file_uploader("Seleziona il file CSV con la colonna SKU", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Anteprima CSV")
        st.dataframe(df, use_container_width=True)

# ---------------- Generazione ----------------
elif selected == "Generazione":
    st.subheader("Configurazione Generazione")
    
    # Editor prompt
    st.markdown("### Modifica Prompt")
    prompt = st_ace(value="Prompt di esempio...", language="python", theme="chrome", height=150)
    
    # Form opzioni generazione
    with st.form("gen_form"):
        langs = st.multiselect("Lingue", ["it", "en", "fr", "de"], default=["it"])
        batch_size = st.number_input("Batch size", min_value=1, value=5)
        submit_gen = st.form_submit_button("Genera Descrizioni")
    
    if submit_gen:
        st.info("Generazione in corso...")
        for i in stqdm(range(batch_size), desc="Processing"):
            pass  # Qui chiami la tua funzione di generazione originale
        st.success("Generazione completata!")

# ---------------- Storico / RAG ----------------
elif selected == "Storico / RAG":
    st.subheader("Visualizza storico / RAG")
    # Qui puoi mostrare lo storico con DataFrame o componenti grafici
    st.write("Qui visualizzi il tuo storico interattivo")

# ---------------- Download ----------------
elif selected == "Download":
    st.subheader("Download CSV / ZIP")
    with io.BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            z.writestr("example.csv", "SKU,Descrizione\n123,Scarpa esempio")
        buffer.seek(0)
        st.download_button("Scarica ZIP", buffer, file_name="descrizioni.zip")

# ---------------- Chat opzionale ----------------
if st.checkbox("Apri Chat con AI"):
    message("Ciao! Chiedimi qualcosa sulle descrizioni generate.", is_user=False)
    user_input = st.text_input("Scrivi qui")
    if user_input:
        message(user_input, is_user=True)
        message("Risposta AI generica...", is_user=False)
