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
from googleapiclient.discovery import build
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

def retrieve_similar(query_row: pd.Series, df: pd.DataFrame, index, k=5, col_weights: Dict[str, float] = {}):
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
        # st.warning(f"⚠️ Errore nel captioning: {str(e)}")
        return ""
        
# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_prompt(row, examples=None, col_display_names=None, image_caption=None):
    fields = []

    if col_display_names is None:
        col_display_names = {col: col for col in row.index}

    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"{label}: {row[col]}")

    product_info = "; ".join(fields)

    # 🧩 Esempi, solo se disponibili
    example_section = ""
    if examples is not None and not examples.empty:
        example_lines = []
        for i, (_, ex) in enumerate(examples.iterrows()):
            desc1 = ex.get("Description", "").strip()
            desc2 = ex.get("Description2", "").strip()
            if desc1 and desc2:
                example = (
                    f"Esempio {i+1}:\n"
                    f"Descrizione lunga: {desc1}\n"
                    f"Descrizione breve: {desc2}"
                )
                example_lines.append(example)
        example_section = "\n\n".join(example_lines)

    # 📄 Prompt finale
    prompt = f"""Scrivi due descrizioni in italiano per una calzatura da vendere online.

Tono richiesto: professionale, user friendly, accattivante, SEO-friendly.
Evita nome prodotto, colore e marchio.

Scheda tecnica: {product_info}
"""
    if image_caption:
        prompt += f"\nAspetto visivo: {image_caption}"
    if example_section:
        prompt += f"\n\nEsempi:\n{example_section}"

    prompt += "\n\nRispondi con:\nDescrizione lunga:\nDescrizione breve:"

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
# 📦 Streamlit UI (RIORGANIZZATA)
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni di Scarpe con RAG")

# 📁 Caricamento dati
with st.sidebar:
    st.header("📥 Caricamento")
    sheet_id = st.secrets["GSHEET_ID"]
    uploaded = st.file_uploader("CSV dei prodotti", type="csv")

    if uploaded:
        df_input = read_csv_auto_encoding(uploaded)
        st.session_state["df_input"] = df_input
         # ✅ Inizializza variabili di stato se non esistono
        if "col_weights" not in st.session_state:
            st.session_state.col_weights = {}
        if "col_display_names" not in st.session_state:
            st.session_state.col_display_names = {}
        if "selected_cols" not in st.session_state:
            st.session_state.selected_cols = []
        if "config_ready" not in st.session_state:
            st.session_state.config_ready = False
        if "generate" not in st.session_state:
            st.session_state.generate = False
        st.success("✅ File caricato con successo!")

# 📊 Anteprima dati
if "df_input" in st.session_state:
    df_input = st.session_state.df_input
    st.subheader("🧾 Anteprima CSV")
    st.dataframe(df_input.head())

    # 🧩 Configurazione colonne
    with st.expander("⚙️ Configura colonne per il prompt", expanded=True):
        st.markdown("### 1. Seleziona colonne")
        available_cols = [col for col in df_input.columns if col not in ["Description", "Description2"]]
        st.session_state.selected_cols = st.multiselect("Colonne da includere nel prompt", options=available_cols, default=[])

        if st.session_state.selected_cols:
            if st.button("▶️ Procedi alla configurazione colonne"):
                st.session_state.config_ready = True

        if st.session_state.get("config_ready"):
            st.markdown("### 2. Configura pesi ed etichette")
            for col in st.session_state.selected_cols:
                st.session_state.col_weights.setdefault(col, 1)
                st.session_state.col_display_names.setdefault(col, col)

                cols = st.columns([2, 3])
                with cols[0]:
                    st.session_state.col_weights[col] = st.slider(
                        f"Peso: {col}", 0, 5, st.session_state.col_weights[col], key=f"peso_{col}"
                    )
                with cols[1]:
                    st.session_state.col_display_names[col] = st.text_input(
                        f"Etichetta: {col}", value=st.session_state.col_display_names[col], key=f"label_{col}"
                    )

    # 🌍 Lingue e parametri
    with st.expander("🌍 Selezione Lingue & Parametri"):
        settings_col1, settings_col2, settings_col3 = st.columns(3)
        with settings_col1:
            selected_labels = st.multiselect(
                "Lingue di output",
                options=list(LANG_LABELS.keys()),
                default=["Italiano"]
            )
            selected_langs = [LANG_LABELS[label] for label in selected_labels]

        with settings_col2:
            use_simili = st.checkbox("Usa descrizioni simili (RAG)", value=True)
            k_simili = 2 if use_simili else 0
            
        with settings_col3:
            use_image = st.checkbox("Usa immagine per descrizioni accurate", value=True)

    # 💵 Stima costi
    if st.button("💰 Stima costi generazione"):
        with st.spinner("Calcolo in corso..."):
            prompts = []
            for _, row in df_input.iterrows():
                simili = pd.DataFrame([])
                image_url = row.get("Image 1", "")
                if use_image:
                    caption = get_blip_caption(row.get("Image 1", "")) if row.get("Image 1", "") else None
                else
                    caption = None
                prompt = build_prompt(row, simili, st.session_state.col_display_names, caption)
                prompts.append(prompt)
                if len(prompts) >= 3:
                    break

            avg_prompt_len = sum(len(p) for p in prompts) / len(prompts)
            avg_prompt_tokens = avg_prompt_len / 4
            output_tokens = 80 * len(df_input) * len(selected_langs)
            total_tokens = avg_prompt_tokens * len(df_input) + output_tokens
            est_cost = total_tokens / 1000 * 0.001

            st.info(f"""
            🧮 Prompt medio: ~{avg_prompt_tokens:.0f} token  
            ✍️ Output stimato per riga: 80 token × {len(selected_langs)} lingue  
            📊 Token totali: ~{int(total_tokens)}  
            💸 **Costo stimato: ${est_cost:.4f}**
            """)

    # 🪄 Generazione descrizioni
    if st.button("🚀 Genera Descrizioni"):
        st.session_state["generate"] = True

    if st.session_state.get("generate"):
        from io import BytesIO
        try:
            # Carica FAISS
            index_df = None
            if sheet_id:
                with st.spinner("📚 Carico storico..."):
                    data_sheet = get_sheet(sheet_id, "it")
                    df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
                    if "faiss_index" not in st.session_state:
                        index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                        st.session_state["faiss_index"] = (index, index_df)
                    else:
                        index, index_df = st.session_state["faiss_index"]

            all_outputs = {lang: [] for lang in selected_langs}
            logs = []

            with st.spinner("✏️ Generazione in corso..."):
                progress_bar = st.progress(0)
                for i, (_, row) in enumerate(df_input.iterrows()):
                    progress_bar.progress((i + 1) / len(df_input))
                    try:
                        simili = (
                            retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)
                            if k_simili > 0 else pd.DataFrame([])
                        )
                        if use_image:
                            caption = get_blip_caption(row.get("Image 1", "")) if row.get("Image 1", "") else None
                        else
                            caption = None
                        prompt = build_prompt(row, simili, st.session_state.col_display_names, caption)
                        gen_output = generate_descriptions(prompt)

                        if "Descrizione breve:" in gen_output:
                            descr_lunga, descr_breve = gen_output.split("Descrizione breve:")
                            descr_lunga = descr_lunga.replace("Descrizione lunga:", "").strip()
                            descr_breve = descr_breve.strip()
                        else:
                            descr_lunga = gen_output.strip()
                            descr_breve = ""

                        base = {**row.to_dict(), "Description": descr_lunga, "Description2": descr_breve}
                        for lang in selected_langs:
                            if lang == "it":
                                all_outputs[lang].append(base)
                            else:
                                trad = base.copy()
                                trad["Description"] = translate_text(base["Description"], target_lang=lang)
                                trad["Description2"] = translate_text(base["Description2"], target_lang=lang)
                                all_outputs[lang].append(trad)

                        logs.append({
                            "sku": row.get("SKU", ""),
                            "status": "OK",
                            "prompt": prompt,
                            "output": gen_output,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                    except Exception as e:
                        logs.append({
                            "sku": row.get("SKU", ""),
                            "status": f"Errore: {str(e)}",
                            "prompt": prompt,
                            "output": "",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        })

            with st.spinner("📤 Salvataggio in Google Sheet..."):
                for lang in selected_langs:
                    df_out = pd.DataFrame(all_outputs[lang])
                    append_to_sheet(sheet_id, lang, df_out)
                for log in logs:
                    append_log(sheet_id, log)

            with st.spinner("📦 Generazione ZIP..."):
                mem_zip = BytesIO()
                with zipfile.ZipFile(mem_zip, "w") as zf:
                    for lang in selected_langs:
                        df_out = pd.DataFrame(all_outputs[lang])
                        df_export = pd.DataFrame({
                            "SKU": df_out.get("SKU", ""),
                            "Descrizione lunga": df_out.get("Description", ""),
                            "Descrizione corta": df_out.get("Description2", "")
                        })
                        zf.writestr(f"descrizioni_{lang}.csv", df_export.to_csv(index=False).encode("utf-8"))
                mem_zip.seek(0)

            st.success("✅ Tutto fatto!")
            st.download_button("📥 Scarica descrizioni (ZIP)", mem_zip, file_name="descrizioni.zip")
            
            st.session_state["generate"] = False

        except Exception as e:
            st.error(f"Errore durante la generazione: {str(e)}")
            st.text(traceback.format_exc())

    # 🔍 Prompt Preview & Benchmark
    with st.expander("🔍 Strumenti di debug & Anteprima"):
        row_index = st.number_input("Indice riga per anteprima", 0, len(df_input) - 1, 0)
        test_row = df_input.iloc[row_index]

        if st.button("💬 Mostra Prompt di Anteprima"):
            with st.spinner("Generazione..."):
                try:
                    if sheet_id:
                        data_sheet = get_sheet(sheet_id, "it")
                        df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
                        if "faiss_index" not in st.session_state:
                            index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                            st.session_state["faiss_index"] = (index, index_df)
                        else:
                            index, index_df = st.session_state["faiss_index"]
                        simili = (
                            retrieve_similar(test_row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)
                            if k_simili > 0 else pd.DataFrame([])
                        )
                    else:
                        simili = pd.DataFrame([])

                    image_url = test_row.get("Image 1", "")
                    if use_image:
                        caption = get_blip_caption(image_url) if image_url else None
                    else
                        caption = None
                    prompt_preview = build_prompt(test_row, simili, st.session_state.col_display_names, caption)
                    st.expander("📄 Prompt generato").code(prompt_preview, language="markdown")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")

        if st.button("🧪 Esegui Benchmark FAISS"):
            with st.spinner("In corso..."):
                benchmark_faiss(df_input, st.session_state.col_weights)
