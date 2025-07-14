import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np
import time
import os
from typing import List, Dict, Any
from google.oauth2 import service_account
import gspread
from io import BytesIO
import zipfile
import chardet
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
import torch
import logging
import traceback
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import asyncio

logging.basicConfig(level=logging.INFO)

LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco"
}
LANG_LABELS = {v.capitalize(): k for k, v in LANG_NAMES.items()}

# ---------------------------
# 🔐 Setup API keys and credentials
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
gsheet_client = gspread.authorize(credentials)

#def get_sheet(sheet_id, tab):
#    try:
#        return gsheet_client.open_by_key(sheet_id).worksheet(tab)
#    except:
#        return gsheet_client.open_by_key(sheet_id).add_worksheet(title=tab, rows="10000", cols="50")

def get_sheet(sheet_id, tab):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    worksheets = spreadsheet.worksheets()
    
    # Confronto case-insensitive per maggiore robustezza
    for ws in worksheets:
        if ws.title.strip().lower() == tab.strip().lower():
            return ws

    # Se non trovato, lo crea
    return spreadsheet.add_worksheet(title=tab, rows="10000", cols="50")

# ---------------------------
# 📦 Embedding & FAISS Setup
# ---------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=st.secrets["HF_TOKEN"])
    return model.to("cpu")

model = load_model()

def embed_texts(texts: List[str], batch_size=32) -> List[List[float]]:
    return model.encode(texts, show_progress_bar=False, batch_size=batch_size).tolist()

def hash_dataframe_and_weights(df: pd.DataFrame, col_weights: Dict[str, float]) -> str:
    df_bytes = pickle.dumps((df.fillna("").astype(str), col_weights))
    return hashlib.md5(df_bytes).hexdigest()

def build_faiss_index(df: pd.DataFrame, col_weights: Dict[str, float], cache_dir="faiss_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hash_dataframe_and_weights(df, col_weights)
    cache_path = os.path.join(cache_dir, f"{cache_key}.index")

    if os.path.exists(cache_path):
        index = faiss.read_index(cache_path)
        return index, df

    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    vectors = embed_texts(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, cache_path)

    return index, df

def retrieve_similar(query_row: pd.Series, df: pd.DataFrame, index, k=2, col_weights: Dict[str, float] = {}):
    parts = []
    for col in df.columns:
        if pd.notna(query_row[col]):
            weight = col_weights.get(col, 1)
            if weight > 0:
                parts.append((f"{col}: {query_row[col]} ") * int(weight))
    query_text = " ".join(parts)

    query_vector = embed_texts([query_text])[0]
    D, I = index.search(np.array([query_vector]).astype("float32"), k)

    # 🔍 DEBUG
    logging.info(f"QUERY TEXT: {query_text[:300]} ...")
    logging.info(f"INDICI trovati: {I[0]}")
    logging.info(f"Distanze: {D[0]}")
    
    return df.iloc[I[0]]

def estimate_embedding_time(df: pd.DataFrame, col_weights: Dict[str, float], sample_size: int = 10) -> float:
    """
    Stima il tempo totale per embeddare tutti i testi del dataframe.
    """
    texts = []
    for _, row in df.head(sample_size).iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    start = time.time()
    _ = embed_texts(texts)
    elapsed = time.time() - start
    avg_time_per_row = elapsed / sample_size
    total_estimated_time = avg_time_per_row * len(df)

    return total_estimated_time

def benchmark_faiss(df, col_weights, query_sample_size=10):
    import os

    st.markdown("### ⏱️ Benchmark FAISS + Embedding")

    start_embed = time.time()
    texts = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" * int(col_weights.get(col, 1))
                 for col in df.columns if pd.notna(row[col])]
        texts.append(" ".join(parts))
    vectors = embed_texts(texts)
    embed_time = time.time() - start_embed

    start_faiss = time.time()
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, "tmp_benchmark.index")
    index_time = time.time() - start_faiss

    index_size = os.path.getsize("tmp_benchmark.index")

    # Test query
    query_times = []
    for i in range(min(query_sample_size, len(df))):
        qtext = texts[i]
        start_q = time.time()
        _ = index.search(np.array([vectors[i]]).astype("float32"), 5)
        query_times.append(time.time() - start_q)

    avg_query_time = sum(query_times) / len(query_times)

    st.write({
        "🚀 Tempo embedding totale (s)": round(embed_time, 2),
        "📄 Tempo medio per riga (ms)": round(embed_time / len(df) * 1000, 2),
        "🏗️ Tempo costruzione FAISS (s)": round(index_time, 2),
        "💾 Dimensione index (KB)": round(index_size / 1024, 1),
        "🔍 Tempo medio query (ms)": round(avg_query_time * 1000, 2),
    })

    os.remove("tmp_benchmark.index")

# ---------------------------
# 🎨 Visual Embedding
# ---------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_auth_token=st.secrets["HF_TOKEN"]
    )
    return processor, model
    
def get_blip_caption(image_url: str) -> str:
    try:
        processor, model = load_blip_model()
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = processor(raw_image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        st.warning(f"⚠️ Errore nel captioning: {str(e)}")
        return ""
        
# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_prompt(row, examples=None, col_display_names=None, image_caption=None):
    fields = []

    if col_display_names is None:
        # fallback per retrocompatibilità
        col_display_names = {col: col for col in row.index}

    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"{label}: {row[col]}")

    

    product_info = "; ".join(fields)

    example_section = "\n\n".join(
        f"""Esempio {i+1}:
    Descrizione lunga: {ex['Description']}
    Descrizione breve: {ex['Description2']}"""
        for i, (_, ex) in enumerate(examples.iterrows())
        if pd.notna(ex.get("Description")) and pd.notna(ex.get("Description2"))
    )

    prompt = f"""Scrivi due descrizioni in italiano per una calzatura da vendere online.

Tono richiesto: professionale, user friendly, accattivante, SEO-friendly.
Evita nome prodotto, colore e marchio.

Scheda tecnica: {product_info}
Aspetto visivo: {image_caption}

Esempi:
{example_section.strip()}


Rispondi con:
Descrizione lunga:
Descrizione breve:"""

    if len(prompt) > 12000:
        st.warning("⚠️ Il prompt generato supera i limiti raccomandati di lunghezza.")
    
    return prompt

def generate_descriptions(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=3000
    )
    return response.choices[0].message.content

# ---------------------------
# 🌍 Traduzione
# ---------------------------
def translate_text(text, target_lang="en"):
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    prompt = f"Traduci il seguente testo in {lang_name}:\n{text}"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ---------------------------
# 📊 Audit Trail & Google Sheets
# ---------------------------
def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")

def overwrite_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    sheet.clear()

    # ✅ Sostituisci NaN con stringa vuota e converti tutto in stringa (safe per JSON)
    df = df.fillna("").astype(str)

    # ✅ Prepara i dati: intestazione + righe
    data = [df.columns.tolist()] + df.values.tolist()

    # ✅ Scrittura nel foglio Google Sheets
    sheet.update(data)

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    for row in values:
        sheet.append_row(row, value_input_option="RAW")
    
# ---------------------------
# Funzioni varie
# ---------------------------
def read_csv_auto_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    uploaded_file.seek(0)  # Rewind after read
    return pd.read_csv(uploaded_file, encoding=encoding)

def estimate_cost(prompt, model="gpt-3.5-turbo"):
    # Conversione caratteri → token (media approssimata: 1 token ≈ 4 caratteri)
    num_chars = len(prompt)
    est_tokens = num_chars // 4

    # Costi approssimativi per 1K token (puoi adattare)
    cost_per_1k = {
        "gpt-3.5-turbo": 0.001,         # input + output stimato
        "gpt-4": 0.03                   # solo input (semplificato)
    }

    cost = (est_tokens / 1000) * cost_per_1k.get(model, 0.001)
    return est_tokens, cost

# ---------------------------
# Async Functions
# ---------------------------
async def generate_description_async(prompt: str) -> str:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"__ERROR__::{str(e)}"

async def generate_descriptions_parallel(prompts: List[str], concurrency: int = 5) -> List[str]:
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(prompt):
        async with semaphore:
            return await generate_description_async(prompt)

    tasks = [sem_task(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
    
# ---------------------------
# 📦 Streamlit UI
# ---------------------------
if uploaded:
    df_input = read_csv_auto_encoding(uploaded)
    st.subheader("📄 Anteprima CSV caricato")
    st.dataframe(df_input.head(), use_container_width=True)

    # ⬇️ Setup session state per configurazione colonne
    for key in ["col_weights", "col_display_names", "selected_cols", "config_ready"]:
        if key not in st.session_state:
            st.session_state[key] = {} if "col" in key else False

    st.divider()

    st.subheader("🌍 Configura lingue di output")

    lang_cols = st.columns(len(LANG_LABELS))
    selected_langs = []
    for idx, (label, lang_code) in enumerate(LANG_LABELS.items()):
        with lang_cols[idx]:
            if st.checkbox(label, value=(lang_code == "IT"), key=f"lang_{lang_code}"):
                selected_langs.append(lang_code)

    st.divider()

    st.subheader("⚙️ Opzioni generazione")

    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        use_simili = st.checkbox("🔁 Usa descrizioni simili dallo storico (RAG)", value=True)

    with col_opt2:
        st.session_state["k_simili"] = 2 if use_simili else 0

    st.markdown("### 🧩 Seleziona colonne da includere nel prompt")

    available_cols = [col for col in df_input.columns if col not in ["Description", "Description2"]]
    st.session_state.selected_cols = st.multiselect(
        "Colonne disponibili", options=available_cols, default=st.session_state.selected_cols
    )

    if st.session_state.selected_cols and st.button("▶️ Procedi alla configurazione colonne"):
        st.session_state.config_ready = True

    if st.session_state.config_ready:
        st.markdown("### 🎛️ Configura pesi e nomi colonne")
        for col in st.session_state.selected_cols:
            if col not in st.session_state.col_weights:
                st.session_state.col_weights[col] = 1
            if col not in st.session_state.col_display_names:
                st.session_state.col_display_names[col] = col

            col1, col2 = st.columns([2, 3])
            with col1:
                st.session_state.col_weights[col] = st.slider(
                    f"Peso per {col}", 0, 5, st.session_state.col_weights[col], key=f"peso_{col}"
                )
            with col2:
                st.session_state.col_display_names[col] = st.text_input(
                    f"Etichetta per {col}", value=st.session_state.col_display_names[col], key=f"label_{col}"
                )

    st.divider()
    st.subheader("🧪 Strumenti")

    test_row_index = st.number_input("Indice riga per anteprima", min_value=0, max_value=len(df_input)-1, value=0)
    test_row = df_input.iloc[test_row_index]

    if st.button("💬 Mostra Prompt di Anteprima"):
        with st.spinner("Generazione prompt..."):
            try:
                simili = pd.DataFrame([])
                if use_simili:
                    data_sheet = get_sheet(sheet_id, "it")
                    df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
                    index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                    simili = retrieve_similar(test_row, index_df, index, k=2, col_weights=st.session_state.col_weights)

                image_url = test_row.get("Image 1", "")
                caption = get_blip_caption(image_url) if image_url else ""
                prompt_preview = build_prompt(test_row, simili, st.session_state.col_display_names, caption)

                with st.expander("📄 Anteprima Prompt"):
                    st.code(prompt_preview)

            except Exception as e:
                st.error(f"Errore nella generazione del prompt: {str(e)}")

    if st.button("⚙️ Esegui Benchmark FAISS"):
        with st.spinner("Eseguo benchmark..."):
            benchmark_faiss(df_input, st.session_state.col_weights)

    if st.button("📊 Stima costi"):
        # (Codice di stima già presente – lo puoi riusare)
        pass

    st.divider()
    st.subheader("🚀 Avvia generazione descrizioni")

    # (Codice già esistente per generazione → lasciato intatto)
