import streamlit as st
import pandas as pd
import openai
import faiss
import os
import time
import json
import asyncio
from typing import List, Dict
from io import BytesIO
from zipfile import ZipFile
from collections import defaultdict
from hashlib import md5

from google.oauth2 import service_account
import gspread
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# --- CONFIGURAZIONI BASE ---
DEFAULT_LANGUAGES = ["it"]
ALL_LANGUAGES = ["it", "en", "fr", "de"]
MODEL = "gpt-3.5-turbo"
WORDS_LONG = 60
WORDS_SHORT = 20
THROTTLE_SECONDS = 1  # ritardo tra chiamate API
FAISS_DIR = "faiss_cache"

# --- CREDENZIALI ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
gs_credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)
gs_client = gspread.authorize(gs_credentials)

# --- INIZIALIZZAZIONE CACHE SESSIONE ---
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame()
if not os.path.exists(FAISS_DIR):
    os.makedirs(FAISS_DIR)

# --- FUNZIONI DI SUPPORTO ---

def get_gsheet_data(spreadsheet_id: str, worksheet: str) -> pd.DataFrame:
    """Legge i dati da un tab di Google Sheets"""
    try:
        sheet = gs_client.open_by_key(spreadsheet_id)
        ws = sheet.worksheet(worksheet)
        return pd.DataFrame(ws.get_all_records())
    except Exception:
        return pd.DataFrame()

def save_to_gsheet(spreadsheet_id: str, worksheet: str, df: pd.DataFrame):
    """Salva un DataFrame su un foglio Google Sheets"""
    sheet = gs_client.open_by_key(spreadsheet_id)
    try:
        ws = sheet.worksheet(worksheet)
        ws.clear()
    except:
        ws = sheet.add_worksheet(title=worksheet, rows=1000, cols=30)
    ws.update([df.columns.values.tolist()] + df.values.tolist())

def append_log(spreadsheet_id: str, log: Dict):
    """Aggiunge una riga al log (tab logs)"""
    logs_df = get_gsheet_data(spreadsheet_id, "logs")
    logs_df = pd.concat([logs_df, pd.DataFrame([log])], ignore_index=True)
    save_to_gsheet(spreadsheet_id, "logs", logs_df)

def generate_prompt(row: Dict, examples: List[Dict]) -> str:
    """Costruisce il prompt da inviare a OpenAI"""
    input_desc = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
    examples_text = "\n\n".join([
        f"Esempio:\n{json.dumps(ex, ensure_ascii=False)}" for ex in examples
    ])
    return (
        f"{examples_text}\n\n"
        f"Genera due descrizioni di scarpe: una lunga ({WORDS_LONG} parole) e una corta ({WORDS_SHORT} parole).\n"
        f"Dati prodotto:\n{input_desc}"
    )

def embed_rows(rows: pd.DataFrame, weights: Dict) -> List[str]:
    """Genera testi pesati per embedding FAISS"""
    return [
        " ".join([str(row[col]) * int(weights.get(col, 1)) for col in rows.columns if pd.notnull(row[col])])
        for _, row in rows.iterrows()
    ]

def faiss_cache_filename(spreadsheet_id: str, weights: Dict) -> str:
    """Genera nome univoco per cache FAISS"""
    key = spreadsheet_id + json.dumps(weights, sort_keys=True)
    return os.path.join(FAISS_DIR, md5(key.encode()).hexdigest() + ".faiss")

def build_faiss(history_df: pd.DataFrame, weights: Dict, cache_key: str):
    """Crea (o carica) un FAISS index con cache su disco"""
    cache_file = faiss_cache_filename(cache_key, weights)
    if os.path.exists(cache_file):
        try:
            return FAISS.load_local(cache_file, OpenAIEmbeddings()), cache_file
        except:
            pass
    if history_df.empty:
        return None, None
    texts = embed_rows(history_df, weights)
    docs = [Document(page_content=row) for row in texts]
    db = FAISS.from_documents(docs, OpenAIEmbeddings())
    db.save_local(cache_file)
    return db, cache_file

def search_similar(row: Dict, db: FAISS, k: int, weights: Dict) -> List[Dict]:
    """Ricerca simili nel FAISS index"""
    if not db:
        return []
    query_text = " ".join([str(row[col]) * int(weights.get(col, 1)) for col in row if row[col]])
    docs = db.similarity_search(query_text, k=k)
    return [{"text": doc.page_content} for doc in docs]

async def call_openai(prompt: str) -> str:
    """Chiamata API OpenAI con throttling"""
    await asyncio.sleep(THROTTLE_SECONDS)
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"ERROR: {e}"

async def generate_descriptions(rows: List[Dict], db: FAISS, weights: Dict) -> List[Dict]:
    """Genera descrizioni per tutte le righe"""
    outputs = []
    for row in rows:
        similar = search_similar(row, db, k=3, weights=weights)
        prompt = generate_prompt(row, similar)
        result = await call_openai(prompt)
        outputs.append({
            **row,
            "description": result.split("\n")[0].strip(),
            "description2": result.split("\n")[1].strip() if "\n" in result else "",
            "prompt_used": prompt,
        })
    return outputs

def estimate_token_cost(num_rows: int, num_langs: int) -> float:
    """Stima token e costo totale"""
    tokens_per_prompt = 500
    total_tokens = tokens_per_prompt * num_rows * num_langs
    return total_tokens, total_tokens / 1000 * 0.001

def translate_text(text: str, lang: str) -> str:
    """Traduzione dummy (può essere sostituita)"""
    return f"[{lang.upper()}] {text}"

def translate_outputs(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Traduce le descrizioni per una lingua specifica"""
    df_translated = df.copy()
    df_translated["description"] = df_translated["description"].apply(lambda x: translate_text(x, lang))
    df_translated["description2"] = df_translated["description2"].apply(lambda x: translate_text(x, lang))
    return df_translated

# --- UI STREAMLIT ---
st.title("🥿 Generatore Descrizioni Scarpe Multilingua")

spreadsheet_id = st.text_input("Google Sheet ID per storico", key="sheet_id")
languages = st.multiselect("Lingue desiderate", ALL_LANGUAGES, default=DEFAULT_LANGUAGES)

uploaded_file = st.file_uploader("Carica il CSV", type="csv")
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write(f"🔍 Trovate {len(input_df)} righe nel file caricato.")

    # --- CONFIGURAZIONE PESI ---
    st.subheader("⚖️ Pesi colonne per RAG")
    col_weights = {}
    for col in input_df.columns:
        if col not in ["description", "description2"]:
            col_weights[col] = st.slider(f"PESO per '{col}'", 0, 5, 1)

    # --- SETUP RAG ---
    if st.button("🔄 Carica storico e costruisci FAISS"):
        history_df = get_gsheet_data(spreadsheet_id, "it")
        st.session_state.history_df = history_df
        db, cache_file = build_faiss(history_df, col_weights, spreadsheet_id)
        st.session_state.faiss_index = db
        if db:
            st.success(f"✅ FAISS creato con {len(history_df)} righe (cache: {cache_file})")
        else:
            st.warning("⚠️ Nessun dato storico disponibile, si procederà senza RAG")

    # --- STIMA COSTO ---
    if st.button("💰 Stima token e costo OpenAI"):
        tokens, cost = estimate_token_cost(len(input_df), len(languages))
        st.info(f"Stima: {tokens} token - Costo: ${cost:.4f} (GPT-3.5)")

    # --- GENERAZIONE E SALVATAGGIO ---
    if st.button("⚙️ Genera descrizioni e salva"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(generate_descriptions(input_df.to_dict(orient="records"), st.session_state.faiss_index, col_weights))
        df_base = pd.DataFrame(results)

        zip_buffer = BytesIO()
        with ZipFile(zip_buffer, "w") as zipf:
            for lang in languages:
                df_lang = translate_outputs(df_base, lang) if lang != "it" else df_base
                save_to_gsheet(spreadsheet_id, lang, df_lang)

                # Audit log
                for _, row in df_lang.iterrows():
                    append_log(spreadsheet_id, {
                        "sku": row.get("SKU", ""),
                        "lang": lang,
                        "success": not row['description'].startswith("ERROR"),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt": row.get("prompt_used", ""),
                        "output": row.get("description", "")
                    })

                zipf.writestr(f"{lang}.csv", df_lang.to_csv(index=False))

        st.download_button("📥 Scarica risultati (ZIP)", zip_buffer.getvalue(), "descrizioni.zip", mime="application/zip")
