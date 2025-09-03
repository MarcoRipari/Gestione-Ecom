import streamlit as st
from st_aggrid import AgGrid
from streamlit_option_menu import option_menu
from stqdm import stqdm
from streamlit_ace import st_ace
from zipfile import ZipFile
import pandas as pd
import io

# ---------- Tema personalizzabile ----------
PRIMARY_COLOR = "#4B7BEC"
SECONDARY_COLOR = "#A29BFE"
BACKGROUND_COLOR = "#F7F9FC"
TEXT_COLOR = "#2D3436"

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")

# ---------- Header ----------
st.markdown(f"""
<div style="background-color:{PRIMARY_COLOR};padding:15px;border-radius:10px">
<h1 style="color:white;text-align:center;">Generatore Descrizioni Calzature</h1>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar Menu ----------
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
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": SECONDARY_COLOR},
            "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
        }
    )

# ---------- Upload CSV ----------
if selected == "Upload CSV":
    st.subheader("Carica il tuo CSV")
    uploaded_file = st.file_uploader("Seleziona il file CSV con la colonna SKU", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.markdown("### Anteprima CSV")
        AgGrid(df, fit_columns_on_grid_load=True)

# ---------- Generazione ----------
elif selected == "Generazione":
    st.subheader("Configurazione Generazione")
    
    # Editor prompt
    st.markdown("### Modifica Prompt")
    prompt = st_ace(value="Prompt di esempio...", language="python", theme="chrome", height=150)
    
    # Opzioni generazione
    with st.form("gen_form"):
        langs = st.multiselect("Lingue", ["it", "en", "fr", "de"], default=["it"])
        batch_size = st.number_input("Batch size", min_value=1, value=5)
        submit_gen = st.form_submit_button("Genera Descrizioni")
    
    if submit_gen:
        st.info("Generazione in corso...")
        for i in stqdm(range(batch_size), desc="Processing"):
            pass  # Qui chiami la tua funzione di generazione originale
        st.success("Generazione completata!")

# ---------- Storico / RAG ----------
elif selected == "Storico / RAG":
    st.subheader("Visualizza storico / RAG")
    # Qui la tua logica attuale per mostrare storico
    st.write("Qui puoi mostrare il tuo storico usando AgGrid o altri componenti")

# ---------- Download ----------
elif selected == "Download":
    st.subheader("Download CSV / ZIP")
    # Generazione file ZIP di esempio
    with io.BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            z.writestr("example.csv", "SKU,Descrizione\n123,Scarpa esempio")
        buffer.seek(0)
        st.download_button("Scarica ZIP", buffer, file_name="descrizioni.zip")
