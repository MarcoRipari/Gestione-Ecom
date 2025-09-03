import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_ace import st_ace
from stqdm import stqdm
from annotated_text import annotated_text
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

# ---------------- Navbar orizzontale ----------------
selected = option_menu(
    menu_title=None,  # Nasconde titolo
    options=["Home", "Upload CSV", "Generazione", "Storico", "Download", "Chat AI"],
    icons=["house", "cloud-upload", "gear", "folder", "download", "chat"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": PRIMARY_COLOR},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "color": "white", "padding": "10px"},
        "nav-link-selected": {"background-color": SECONDARY_COLOR, "color": "white"},
    }
)

# ---------------- Sezioni ----------------
if selected == "Home":
    st.markdown(f"""
    <div style="background-color:{CARD_BG};padding:50px;border-radius:15px;
                box-shadow:0 6px 12px rgba(0,0,0,0.1);text-align:center;">
        <h1 style="color:{PRIMARY_COLOR};margin-bottom:10px;">Generatore Descrizioni Calzature</h1>
        <p style="color:{TEXT_COLOR};font-size:18px;">Benvenuto! Usa il menu in alto per navigare.</p>
    </div>
    """, unsafe_allow_html=True)

elif selected == "Upload CSV":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:25px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.header("📂 Carica il tuo CSV")
    uploaded_file = st.file_uploader("Trascina qui il file CSV con la colonna SKU", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File caricato con successo!")
        st.subheader("Anteprima")
        st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Generazione":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:25px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.header("⚙️ Generazione Descrizioni")
    
    st.subheader("Prompt Editor")
    prompt = st_ace(value="Scrivi qui il prompt di esempio...", language="markdown", theme="chrome", height=200)
    
    with st.form("gen_form"):
        langs = st.multiselect("Lingue disponibili", ["it", "en", "fr", "de"], default=["it"])
        batch_size = st.slider("Batch size", 1, 50, 5)
        st.info("Suggerimento: batch più grandi possono aumentare i tempi di elaborazione ⏳")
        submit_gen = st.form_submit_button("🚀 Avvia Generazione")
    
    if submit_gen:
        st.warning("Generazione in corso...")
        for _ in stqdm(range(batch_size), desc="Processing"):
            pass
        st.success("✨ Generazione completata!")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Storico":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:25px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.header("📊 Storico & RAG")
    
    # Esempio grafico
    demo_df = pd.DataFrame({"Lingua": ["it", "en", "fr"], "Descrizioni": [120, 95, 60]})
    fig = px.bar(demo_df, x="Lingua", y="Descrizioni", title="Distribuzione descrizioni per lingua")
    st.plotly_chart(fig, use_container_width=True)
    
    annotated_text(("SKU123", "Scarpa elegante", "#faa"), " generata con successo.")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Download":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:25px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.header("⬇️ Download CSV / ZIP")
    
    with io.BytesIO() as buffer:
        with ZipFile(buffer, "w") as z:
            z.writestr("example.csv", "SKU,Descrizione\n123,Scarpa esempio")
        buffer.seek(0)
        st.download_button("📦 Scarica ZIP", buffer, file_name="descrizioni.zip")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "Chat AI":
    st.markdown(f"<div style='background-color:{CARD_BG};padding:25px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1)'>", unsafe_allow_html=True)
    st.header("💬 Chat con AI")
    message("Ciao! Vuoi un feedback sulle tue descrizioni?", is_user=False)
    user_input = st.text_input("Scrivi qui")
    if user_input:
        message(user_input, is_user=True)
        message("Risposta AI generica...", is_user=False)
    st.markdown("</div>", unsafe_allow_html=True)
