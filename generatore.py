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

logging.basicConfig(level=logging.INFO)

# ---------------------------
# 🔐 Setup API keys and credentials
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
gsheet_client = gspread.authorize(credentials)

def get_sheet(sheet_id, tab):
    try:
        return gsheet_client.open_by_key(sheet_id).worksheet(tab)
    except:
        return gsheet_client.open_by_key(sheet_id).add_worksheet(title=tab, rows="10000", cols="50")

# ---------------------------
# 📦 Embedding & FAISS Setup
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # leggero e veloce

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
    logging.info("QUERY TEXT:", query_text[:300], "...")
    logging.info("INDICI trovati:", I[0])
    logging.info("Distanze:", D[0])
    
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
# 🧠 Prompting e Generazione
# ---------------------------
def build_prompt(row, examples=None, col_display_names=None):
    fields = []
    for col in row.index:
        if pd.notna(row[col]) and col.lower() not in ["sku", "description", "description2"]:
            label = col_display_names.get(col, col) if col_display_names else col
            fields.append(f"{label}: {row[col]}")
    
    product_info = "; ".join(fields)

    example_section = ""
    if examples is not None and not examples.empty:
        ex = examples.iloc[0]
        example_section = f"- {ex['Description']}"

    prompt = f"""Scrivi due descrizioni in italiano per una calzatura da vendere online.

Tono richiesto: professionale, user friendly, accattivante, SEO-friendly.
Evita nome prodotto, colore e marchio.

Esempio:
{example_section.strip()}

Scheda tecnica: {product_info}

Rispondi con:
Descrizione lunga:
Descrizione breve:"""
    return prompt

def generate_descriptions(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# ---------------------------
# 🌍 Traduzione
# ---------------------------
def translate_text(text, target_lang="en"):
    prompt = f"Traduci il seguente testo in {target_lang}:\n{text}"
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

def save_to_sheet(sheet_id, tab, df):
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
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni di Scarpe con RAG")

# sheet_id = st.text_input("Google Sheet ID dello storico", key="sheet")
sheet_id = st.secrets["GSHEET_ID"]
selected_langs = st.multiselect("Seleziona lingue di output", ["it", "en", "fr", "de"], default=["it"])
uploaded = st.file_uploader("Carica il CSV dei prodotti", type="csv")

# Configurazione pesi colonne per RAG
if uploaded:
    #df_input = pd.read_csv(uploaded)
    df_input = read_csv_auto_encoding(uploaded)
    st.dataframe(df_input.head())
    col_weights = {}
    col_display_names = {}
    st.markdown("### ⚙️ Configura i pesi e i nomi delle colonne")
    
    for col in df_input.columns:
        if col not in ["Description", "Description2"]:
            cols = st.columns([2, 3])
            with cols[0]:
                col_weights[col] = st.slider(f"Peso: {col}", 0, 5, 1, key=f"peso_{col}")
            with cols[1]:
                col_display_names[col] = st.text_input(f"Etichetta: {col}", value=col, key=f"label_{col}")


    row_index = st.number_input("🔢 Indice riga per anteprima prompt", 0, len(df_input)-1, 0)
    test_row = df_input.iloc[row_index]

    if st.button("Esegui Benchmark FAISS"):
        benchmark_faiss(df_input, col_weights)

    # Stimo il costo del token con RAG
    if st.button("💬 Mostra Prompt di Anteprima"):
        with st.spinner("Genero il prompt..."):

            try:
                if sheet_id:
                    # Carica storico ed esegui FAISS
                    data_sheet = get_sheet(sheet_id, "it")
                    df_storico = pd.DataFrame(data_sheet.get_all_records())
                    df_storico = df_storico.tail(500)  # usa solo gli ultimi 500
                    index, index_df = build_faiss_index(df_storico, col_weights)
                    simili = retrieve_similar(test_row, index_df, index, k=1, col_weights=col_weights)
                else:
                    simili = pd.DataFrame([])
    
                prompt_preview = build_prompt(test_row, simili, col_display_names)  # 🔧 fix qui
                prompt_tokens = len(prompt_preview) / 4  # stima token
    
                with st.expander("📄 Anteprima prompt generato"):
                    st.code(prompt_preview, language="markdown")
    
            except Exception as e:
                st.error(f"Errore nella generazione del prompt: {str(e)}")
            
    if st.button("Stima costi"):
        # Calcolo prompt medio sui primi 3 record
        prompts = []
        for _, row in df_input.iterrows():
            simili = pd.DataFrame([])  # niente RAG per la stima
            prompt = build_prompt(row, simili, col_display_names)
            prompts.append(prompt)
            if len(prompts) >= 3:
                break

        # Stimiamo i token: 1 token ≈ 4 caratteri
        avg_prompt_len_chars = sum(len(p) for p in prompts) / len(prompts)
        avg_prompt_tokens = avg_prompt_len_chars / 4

        # Output: descrizione lunga + breve (stima in token)
        output_tokens_per_row = 60 + 20  # circa 80 token per lingua

        # Calcolo finale
        num_rows = len(df_input)
        num_langs = len(selected_langs)
        total_input_tokens = num_rows * avg_prompt_tokens
        total_output_tokens = num_rows * output_tokens_per_row * num_langs
        total_tokens = total_input_tokens + total_output_tokens

        est_cost = total_tokens / 1000 * 0.001  # gpt-3.5-turbo

        st.info(f"""
        Prompt medio: ~{avg_prompt_tokens:.0f} token  
        Output stimato per riga: {output_tokens_per_row} token × {num_langs} lingue  
        Token totali stimati: ~{int(total_tokens)}  
        **Costo stimato: ${est_cost:.4f}**
        """)

    if st.button("Genera Descrizioni"):
        index_df = None
        if sheet_id:
            try:
                data_sheet = get_sheet(sheet_id, "it")
                #df_storico = pd.DataFrame(data_sheet.get_all_records())
                data_sheet = get_sheet(sheet_id, "it")
                df_storico = pd.DataFrame(data_sheet.get_all_records())
                df_storico = df_storico.tail(500)  # usa solo gli ultimi 500
                index, index_df = build_faiss_index(df_storico, col_weights)
            except:
                index = None

        all_outputs = {lang: [] for lang in selected_langs}
        logs = []

        prompt = ""  # <- inizializza la variabile fuori dal try
        
        progress_bar = st.progress(0)
        total = len(df_input)
        
        #for _, row in df_input.iterrows():
        for i, (_, row) in enumerate(df_input.iterrows()):
            progress_bar.progress((i + 1) / total)
            try:
                if index_df is not None:
                    simili = retrieve_similar(row, index_df, index, k=1, col_weights=col_weights)
                else:
                    simili = pd.DataFrame([])

                prompt = build_prompt(row, simili, col_display_names)
                gen_output = generate_descriptions(prompt)

                if "Descrizione breve:" in gen_output:
                    descr_lunga, descr_breve = gen_output.split("Descrizione breve:")
                    descr_lunga = descr_lunga.replace("Descrizione lunga:", "").strip()
                    descr_breve = descr_breve.strip()
                else:
                    # fallback se il modello non segue il formato atteso
                    descr_lunga = gen_output.strip()
                    descr_breve = ""
                    
                base = {
                    **row.to_dict(),
                    #"Description": descr_lunga.strip().replace("Descrizione lunga:", "").strip(),
                    #"Description2": descr_breve.strip()
                    "Description": descr_lunga,
                    "Description2": descr_breve
                }

                for lang in selected_langs:
                    if lang == "it":
                        all_outputs[lang].append(base)
                    else:
                        trad_lunga = translate_text(base["Description"], target_lang=lang)
                        trad_breve = translate_text(base["Description2"], target_lang=lang)
                        trad = base.copy()
                        trad["Description"] = trad_lunga
                        trad["Description2"] = trad_breve
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

        # Salvataggio su Google Sheets
        for lang in selected_langs:
            df_out = pd.DataFrame(all_outputs[lang])
            # save_to_sheet(sheet_id, lang, df_out)
            append_to_sheet(sheet_id, lang, df_out)

        for log in logs:
            append_log(sheet_id, log)

        # Preparazione ZIP
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w") as zf:
            for lang in selected_langs:
                df_out = pd.DataFrame(all_outputs[lang])

                # Riorganizza e rinomina le colonne
                df_export = pd.DataFrame()
                df_export["SKU"] = df_out.get("SKU", "")
                df_export["Descrizione lunga"] = df_out.get("Description", "")
                df_export["Descrizione corta"] = df_out.get("Description2", "")

                csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                zf.writestr(f"descrizioni_{lang}.csv", csv_bytes)
        mem_zip.seek(0)
        st.success("✅ Generazione completata con successo!")
        st.download_button("📥 Scarica CSV (ZIP)", mem_zip, file_name="descrizioni.zip")
