import streamlit as st
import pandas as pd
#from config import (
#    LANG_LABELS, LANG_NAMES, st_init, st_config,
#    st_language_ui, st_column_ui, st_cost_estimation_ui,
#    st_prompt_preview_ui, st_download_zip
#)
from gdrive import (
    get_sheet, append_log, append_to_sheet, read_csv_auto_encoding
)
from retrieval import (
    build_faiss_index, retrieve_similar, benchmark_faiss
)
from prompting import (
    build_unified_prompt, calcola_tokens, get_blip_caption
)
from execution import run_generation_async

# ---------------------------
# 🎯 INIT
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni di Scarpe con RAG")

# ---------------------------
# 📥 Caricamento file
# ---------------------------
with st.sidebar:
    DEBUG = st.checkbox("Debug")
    st.header("📥 Caricamento")
    sheet_id = st.secrets["GSHEET_ID"]
    uploaded = st.file_uploader("CSV dei prodotti", type="csv")
    if uploaded:
        df_input = read_csv_auto_encoding(uploaded)
        st.session_state["df_input"] = df_input
        st_init()
        st.success("✅ File caricato con successo!")

# ---------------------------
# ⚙️ UI e logica principali
# ---------------------------
if "df_input" in st.session_state:
    df_input = st.session_state["df_input"]

    # 🧾 Anteprima CSV
    st.subheader("🧾 Anteprima CSV")
    st.dataframe(df_input.head())

    # ⚙️ Selezione colonne
    st_column_ui(df_input)

    if st.session_state.get("config_ready"):
        # 🌍 Lingue + Parametri
        marchio, selected_langs, selected_tones, k_simili, use_image, desc_lunga_length, desc_breve_length = st_language_ui()

        # 💸 Stima costi
        st_cost_estimation_ui(
            df_input=df_input,
            selected_langs=selected_langs,
            selected_tones=selected_tones,
            desc_lunga_length=desc_lunga_length,
            desc_breve_length=desc_breve_length,
            k_simili=k_simili,
            use_image=use_image
        )

        # 🚀 Generazione
        if st.button("🚀 Genera Descrizioni"):
            st.session_state["generate"] = True

        if st.session_state.get("generate"):
            run_generation_async(
                df_input=df_input,
                sheet_id=sheet_id,
                marchio=marchio,
                selected_langs=selected_langs,
                selected_tones=selected_tones,
                desc_lunga_length=desc_lunga_length,
                desc_breve_length=desc_breve_length,
                k_simili=k_simili,
                use_image=use_image,
                DEBUG=DEBUG
            )
            st.session_state["generate"] = False

        # 🧪 Debug Prompt & Benchmark
        st_prompt_preview_ui(
            df_input=df_input,
            selected_langs=selected_langs,
            marchio=marchio,
            k_simili=k_simili,
            use_image=use_image
        )

        if st.button("🧪 Esegui Benchmark FAISS"):
            benchmark_faiss(df_input, st.session_state.col_weights)
