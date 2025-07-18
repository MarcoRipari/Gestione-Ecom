import streamlit as st
import pandas as pd
import traceback
from io import BytesIO
import zipfile
import time
import json

# 🌐 Moduli locali
from config import LANG_NAMES, LANG_LABELS, DEFAULT_COLUMNS, AVAILABLE_TONES, DESCRIPTION_LENGTHS, EXCLUDED_COLUMNS, LOG_SHEET, STORICO_PREFIX
from gdrive import read_csv_auto_encoding, get_sheet, append_log, append_to_sheet, overwrite_sheet
from retrieval import build_faiss_index, retrieve_similar, get_blip_caption, benchmark_faiss, estimate_embedding_time
from prompting import build_unified_prompt, calcola_tokens, generate_all_prompts

# ---------------------------
# 🌐 UI Streamlit
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni di Scarpe con RAG")

# 📂 Caricamento file CSV
with st.sidebar:
    DEBUG = st.checkbox("Debug")
    st.header("📅 Caricamento")
    sheet_id = st.secrets["GSHEET_ID"]
    uploaded = st.file_uploader("CSV dei prodotti", type="csv")

    if uploaded:
        df_input = read_csv_auto_encoding(uploaded)
        st.session_state["df_input"] = df_input

        # ✅ Stato iniziale
        for k in ["col_weights", "col_display_names", "selected_cols", "config_ready", "generate"]:
            st.session_state.setdefault(k, {} if "weights" in k or "display" in k else False)
        st.success("✅ File caricato con successo!")

# 📊 Anteprima
if "df_input" in st.session_state:
    df_input = st.session_state.df_input
    st.subheader("📜 Anteprima CSV")
    st.dataframe(df_input.head())

    # ⚙️ Configurazione colonne
    with st.expander("⚙️ Configura colonne", expanded=True):
        st.markdown("### 1. Seleziona colonne")
        available_cols = [col for col in df_input.columns if col not in EXCLUDED_COLUMNS]

        def_column = [col for col in DEFAULT_COLUMNS if col in df_input.columns]
        st.session_state.selected_cols = st.multiselect("Colonne da includere", options=available_cols, default=def_column)

        if st.session_state.selected_cols and st.button("▶️ Procedi alla configurazione colonne"):
            st.session_state.config_ready = True

        if st.session_state.get("config_ready"):
            st.markdown("### 2. Configura pesi ed etichette")
            for col in st.session_state.selected_cols:
                st.session_state.col_weights.setdefault(col, 1)
                st.session_state.col_display_names.setdefault(col, col)
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.session_state.col_weights[col] = st.slider(f"Peso: {col}", 0, 5, st.session_state.col_weights[col], key=f"peso_{col}")
                with c2:
                    st.session_state.col_display_names[col] = st.text_input(f"Etichetta: {col}", st.session_state.col_display_names[col], key=f"label_{col}")

    # 🌍 Lingue e parametri
    with st.expander("🌍 Lingue & Parametri"):
        col1, col2, col3 = st.columns(3)
        with col1:
            marchio = st.radio("Marchio", ["NAT", "FAL", "VB", "FM", "WZ", "CC"])
            use_simili = st.checkbox("Usa descrizioni simili", value=True)
            k_simili = 2 if use_simili else 0
            use_image = st.checkbox("Usa immagine", value=True)

        with col2:
            selected_labels = st.multiselect("Lingue output", options=list(LANG_LABELS.keys()), default=["Italiano", "Inglese", "Francese", "Tedesco"])
            selected_langs = [LANG_LABELS[label] for label in selected_labels]
            selected_tones = st.multiselect("Tono", AVAILABLE_TONES, default=["professionale", "user friendly", "SEO-friendly"])

        with col3:
            desc_lunga_length = st.selectbox("Lunghezza descrizione lunga", DESCRIPTION_LENGTHS, index=5)
            desc_breve_length = st.selectbox("Lunghezza descrizione breve", DESCRIPTION_LENGTHS, index=1)

    # 💵 Stima token/costo
    if st.button("💰 Stima costi generazione"):
        token_est, cost_est, prompt = calcola_tokens(
            df_input=df_input,
            col_display_names=st.session_state.col_display_names,
            selected_langs=selected_langs,
            selected_tones=selected_tones,
            desc_lunga_length=desc_lunga_length,
            desc_breve_length=desc_breve_length,
            k_simili=k_simili,
            use_image=use_image,
            faiss_index=st.session_state.get("faiss_index"),
            DEBUG=True
        )
        if token_est:
            st.info(f"\n📊 Token stimati: ~{token_est}\n💸 Costo stimato per riga: ${cost_est:.6f}")

    # 🪄 Avvio generazione
    if st.button("🚀 Genera Descrizioni"):
        st.session_state.generate = True

# ✅ L'intera logica async, generazione, salvataggio, download è contenuta in un modulo separato
if st.session_state.get("generate"):
    from execution import execute_generation_async
    execute_generation_async(
        df_input=df_input,
        sheet_id=sheet_id,
        marchio=marchio,
        selected_langs=selected_langs,
        selected_tones=selected_tones,
        desc_lunga_length=desc_lunga_length,
        desc_breve_length=desc_breve_length,
        use_image=use_image,
        k_simili=k_simili,
        DEBUG=DEBUG
    )
