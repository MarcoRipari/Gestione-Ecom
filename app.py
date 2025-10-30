import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import openai
import faiss
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import os
from io import BytesIO
import zipfile
import chardet
import hashlib
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pandas as pd
import torch
import logging
import traceback
from transformers import BlipProcessor, BlipForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, pipeline
from PIL import Image
import requests
import asyncio
import aiohttp
import json
from openai import AsyncOpenAI
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_fixed
import dropbox
from dropbox.files import WriteMode
import base64
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import re
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
from reportlab.lib.pagesizes import landscape
import html
import io
from dateutil import parser
from dateutil.tz import tzlocal
import locale
from zoneinfo import ZoneInfo
import pdfplumber
import PyPDF2
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import zipfile
from google.oauth2 import service_account
import gspread
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.http import MediaInMemoryUpload
from gspread.utils import rowcol_to_a1
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
import gspread.utils
from googleapiclient.discovery import build
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from collections import deque
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

import viste

from functions.supabase_creds import *
from functions.auth_system import *

logging.basicConfig(level=logging.INFO)

LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco",
    "ES": "spagnolo"
}
LANG_LABELS = {v.capitalize(): k for k, v in LANG_NAMES.items()}


# ---------------------------
# GLOBAL VARS
# ---------------------------
desc_sheet_id = st.secrets['DESC_GSHEET_ID']
foto_sheet_id = st.secrets['FOTO_GSHEET_ID']
anagrafica_sheet_id = st.secrets['ANAGRAFICA_GSHEET_ID']
ferie_sheet_id = st.secrets['FERIE_GSHEET_ID']
ordini_sheet_id = st.secrets['ORDINI_GSHEET_ID']
MISTRAL_API_KEY = st.secrets['MISTRAL_API_KEY']
OPENROUTER_API_KEY = st.secrets['OPENROUTER_API_KEY']

# ---------------------------
# 🔐 Setup API keys and credentials
# ---------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

def check_openai_key():
    try:
        openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True
    except Exception as e:  
        msg = str(e).lower()
        return False


# ---------------------------
# Github Repo
# ---------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]  # Puoi usare st.secrets in Streamlit Cloud
git_headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
git_data = {
        "ref": "main",
        "inputs": {}
    }
def get_last_run(owner, repo, file):
    filename = f"{file}.yml"
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{filename}/runs"
    response = requests.get(url, headers=git_headers)
    if response.status_code == 200:
        runs = response.json()["workflow_runs"]
        if runs:
            return runs[0]  # L’ultima esecuzione
    return None

def run_workflow(owner, repo, file):
    filename = f"{file}.yml"
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{filename}/dispatches"

    response = requests.post(url, headers=git_headers, json=git_data)
    return response

def get_workflow_logs(owner, repo, run_id, artifact_name="output.txt"):
    # 1. Ottieni URL di download artifact
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    r = requests.get(url, headers=git_headers)
    artifacts = r.json()["artifacts"]
    if not artifacts:
        return "Nessun artifact trovato."

    download_url = artifacts[0]["archive_download_url"]

    # 2. Scarica il file zip
    r = requests.get(download_url, headers=git_headers)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    # 3. Leggi file .txt dentro lo zip
    with z.open(artifact_name) as f:
        content = f.read().decode("utf-8")
        return content
    
def workflow(owner, repo, file, interval=5, timeout=120):
    run = run_workflow(owner, repo, file)
    if run.status_code != 204:
        return st.error(f"❌ Errore: {run.status_code} - {run.text}")
        
    time.sleep(10)
    
    run = get_last_run(owner, repo, file)
    run_id = run["id"]

    with st.spinner("Controllo in corso..."):
        start_time = time.time()
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}"
            response = requests.get(url, headers=git_headers)
            if response.status_code == 200:
                data = response.json()
                status = data["status"]
                conclusion = data["conclusion"]
                if status == "completed":
                    logs = get_workflow_logs(owner, repo, run_id)
                    st.write(logs)
                    return
            if time.time() - start_time > timeout:
                return "timeout"
            time.sleep(interval)

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
    #processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    #model = BlipForConditionalGeneration.from_pretrained(
    #    "Salesforce/blip-image-captioning-base",
    #    use_auth_token=st.secrets["HF_TOKEN"]
    #)
    # Carica modello, tokenizer e processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model
    
def get_blip_caption(image_url: str) -> str:
    try:
        processor, model = load_blip_model()

        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        prompt = "Raccogli informazioni descrittive della scarpa nella foto"

        inputs = processor(raw_image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=30)
        
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        # st.warning(f"⚠️ Errore nel captioning: {str(e)}")
        return ""



def get_blip_caption_new(image_url: str) -> str:
    try:
        processor, model = load_blip_model()
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        caption = f"Errore: {e}"
        
    return caption
    
# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_unified_prompt(row, col_display_names, selected_langs, image_caption=None, simili=None):
    # Costruzione scheda tecnica
    fields = []
    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"- {label}: {row[col]}")
    product_info = "\n".join(fields)

    # Elenco lingue in stringa
    lang_list = ", ".join([LANG_NAMES.get(lang, lang) for lang in selected_langs])

    # Caption immagine
    image_line = f"\nAspetto visivo: {image_caption}" if image_caption else ""

    # Descrizioni simili
    sim_text = ""
    if simili is not None and not simili.empty:
        sim_lines = []
        for _, ex in simili.iterrows():
            dl = ex.get("Description", "").strip()
            db = ex.get("Description2", "").strip()
            if dl and db:
                sim_lines.append(f"- {dl}\n  {db}")
        if sim_lines:
            sim_text = "\nDescrizioni simili:\n" + "\n".join(sim_lines)

    # Prompt finale
    prompt = f"""Scrivi due descrizioni coerente con le INFO ARTICOLO per una calzatura da vendere online (e-commerce) in ciascuna delle seguenti lingue: {lang_list}.

>>> GUIDA STILE E LINGUAGGIO
- Tono: {", ".join(selected_tones)}
- Lingua: adatta al paese target
- Mai usare: Codice, Nome, Marca, Colore
- Utilizza esclusivamente il tipo di calzatura passato nelle info articoli
- Non usare generi o età (es. maschile/femminile, bambino/bambina)
- Evita le percentuali materiali
- Verifica la correttezza della descrizione rispetto alla stagione tra le info articolo
- Non usare **alcuna formattazione Markdown** nell'output

>>> PAROLE DA EVITARE (anche implicite)
- velcro → usa "strappo"
- velluto → usa "velour" o "suede"
- primavera, estate, autunno, inverno (e derivati)

>>> ESEMPI DI ERRORI DA EVITARE
❌ "velluto" → ✅ "velour" o "suede"
❌ "primaverile" → ✅ descrizione neutra sulla stagione
❌ "per bambina" → ✅ descrizione neutra

>>> REGOLE
- desc_lunga: {desc_lunga_length} parole → enfasi su comfort, materiali, utilizzo
- desc_breve: {desc_breve_length} parole → adatta a social media o schede prodotto rapide
- Output JSON: {{"it":{{"desc_lunga":"...","desc_breve":"..."}}, "en":{{...}}, "fr":{{...}}, "de":{{...}}}}

>>> INFO ARTICOLO
{product_info}
{image_line}

{sim_text}

Dopo aver generato le descrizioni, rileggi e correggi eventuali errori grammaticali o di genere **prima** di produrre l'output finale JSON.

>>> CONTROLLO FINALE
Controlla attentamente che le descrizioni:
- rispettino tutte le regole fornite (parole vietate, formato, tono, ecc.)
- non contengano errori grammaticali, di concordanza o di traduzione in nessuna lingua
- in italiano, controlla sempre il genere e il numero dei sostantivi (es. "questi sandali", non "queste sandali")
- se trovi errori di grammatica, rigenera o correggi la frase **prima di fornire l'output finale**
- fornisci l'output finale **solo dopo** aver verificato che sia grammaticalmente e stilisticamente corretto in tutte le lingue
"""
    return prompt

client = AsyncOpenAI(api_key=openai.api_key)

# Configurazione per il piano gratuito
RPS_LIMIT = 1  # 1 richiesta al secondo per il piano gratuito
RPS_LIMIT_DEEPSEEK = 20 # Prova per deepseek
MAX_RETRIES = 3  # Numero massimo di tentativi per ogni richiesta
DELAY_BETWEEN_REQUESTS = 1  # 1 secondo tra una richiesta e l'altra
TOKEN_WINDOW = deque()  # (timestamp, token_count)
MAX_TOKENS_PER_MINUTE = 500000

def check_token_limit(tokens: int) -> bool:
    current_time = time.time()
    # Rimuovi i record più vecchi di 60 secondi
    while TOKEN_WINDOW and current_time - TOKEN_WINDOW[0][0] > 60:
        TOKEN_WINDOW.popleft()

    total_tokens = sum(count for _, count in TOKEN_WINDOW)
    if total_tokens + tokens > MAX_TOKENS_PER_MINUTE:
        return False  # Limite superato

    TOKEN_WINDOW.append((current_time, tokens))
    return True

# -----------------------------
# RATE LIMITER (20 richieste/min)
# -----------------------------
MAX_REQUESTS_PER_MIN = RPS_LIMIT_DEEPSEEK
request_times = deque()

async def rate_limiter():
    """Assicura che non vengano fatte più di MAX_REQUESTS_PER_MIN richieste in 60s."""
    now = time.time()
    # Rimuove timestamp più vecchi di 60s
    while request_times and now - request_times[0] > 60:
        request_times.popleft()

    if len(request_times) >= MAX_REQUESTS_PER_MIN:
        sleep_for = 60 - (now - request_times[0])
        st.warning(f"⏳ Rate limit raggiunto. Attendo {sleep_for:.1f}s...")
        await asyncio.sleep(sleep_for)
        # Dopo l'attesa, aggiorna il deque
        now = time.time()
        while request_times and now - request_times[0] > 60:
            request_times.popleft()

    # Registra questa richiesta
    request_times.append(time.time())


async def async_generate_description(prompt: str, idx: int, use_model: str):
    if len(prompt) < 50:
        return idx, {"result": prompt, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        
    try:
        if use_model == "gpt-5-nano":
            response = await client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=3000
            )
        else:
            response = await client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
        content = response.choices[0].message.content
        usage = response.usage
        data = json.loads(content)
        return idx, {"result": data, "usage": usage.model_dump()}
    except Exception as e:
        return idx, {"error": str(e)}

async def async_generate_description_mistral(
    session: aiohttp.ClientSession,
    prompt: str,
    idx: int,
    use_model: str,
    semaphore: asyncio.Semaphore
):
    if len(prompt) < 50:
        return idx, {"result": prompt, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    if use_model not in ["mistral-medium", "deepseek-chimera"]:
        # Gestione per altri modelli (es. OpenAI)
        try:
            response = await client.chat.completions.create(
                model=use_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            content = response.choices[0].message.content
            usage = response.usage
            data = json.loads(content)
            return idx, {"result": data, "usage": usage.model_dump()}
        except Exception as e:
            return idx, {"error": str(e)}
    
    if use_model == "deepseek-chimera":
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:  # Limita richieste concorrenti
                    await rate_limiter()  # ⏱️ Controlla limite 20/min

                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                    }

                    data = {
                        "model": "tngtech/deepseek-r1t2-chimera:free",
                        "messages": [{"role": "user", "content": prompt}],
                    }

                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 429:
                            st.warning("🚫 Rate limit API raggiunto. Attendo 60s...")
                            await asyncio.sleep(60)
                            continue

                        if response.status != 200:
                            error_msg = await response.text()
                            st.write(f"{error_msg}")
                            raise Exception(f"API Error: {error_msg}")

                        response_json = await response.json()
                        content = response_json["choices"][0]["message"]["content"]
                        content = content.replace("**", "")
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)

                        if json_match:
                            content = json.loads(json_match.group(0))
                        else:
                            raise Exception("No valid JSON found in response")

                        usage = response_json.get("usage", {})
                        return idx, {"result": content, "usage": usage}

            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return idx, {"error": f"Failed after {MAX_RETRIES} attempts: {str(e)}"}
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))  # Attesa esponenziale

    
    # Gestione per Mistral
    if use_model == "mistral-medium":
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:  # Limita le richieste concorrenti
                    headers = {
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": use_model,
                        "messages": [{"role": "user", "content": prompt}]
                    }
    
                    async with session.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data) as response:
                        if response.status != 200:
                            error_msg = await response.text()
                            st.write(f"{error_msg}")
                            raise Exception(f"API Error: {error_msg}")
    
                        response_json = await response.json()
                        content = response_json["choices"][0]["message"]["content"]
                        content = content.replace("**", "")  # Rimuovi eventuali **
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
    
                        if json_match:
                            content = json.loads(json_match.group(0))
                        else:
                            raise Exception("No valid JSON found in response")
    
                        usage = response_json.get("usage", {})
                        total_tokens = usage.get("total_tokens", 0)

                        # Controlla il limite dei token al minuto
                        if not check_token_limit(total_tokens):
                            raise Exception("Token limit per minute exceeded")
                            
                        return idx, {"result": content, "usage": usage}
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return idx, {"error": f"Failed after {MAX_RETRIES} attempts: {str(e)}"}
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))  # Attesa esponenziale
            


async def generate_all_prompts(prompts: list[str], model: str) -> dict:
    tasks = [async_generate_description(prompt, idx, model) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)

async def generate_all_prompts_mistral(prompts: list[str], model: str) -> dict:
    semaphore = asyncio.Semaphore(RPS_LIMIT)  # Limita a 1 richiesta al secondo

    async with aiohttp.ClientSession() as session:
        tasks = [
            async_generate_description_mistral(session, prompt, idx, model, semaphore)
            for idx, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

async def generate_all_prompts_deepseek(prompts: list[str], model: str) -> dict:
    semaphore = asyncio.Semaphore(5)  # Limita a X richieste al secondo

    async with aiohttp.ClientSession() as session:
        tasks = [
            async_generate_description_mistral(session, prompt, idx, model, semaphore)
            for idx, prompt in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

def calcola_tokens(df_input, col_display_names, selected_langs, selected_tones, desc_lunga_length, desc_breve_length, k_simili, use_image, faiss_index, DEBUG=False):
    if df_input.empty:
        return None, None, "❌ Il CSV è vuoto"

    row = df_input.iloc[0]

    simili = pd.DataFrame([])
    if k_simili > 0 and faiss_index:
        index, index_df = faiss_index
        simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)
        
    if use_image:
        try:
            sku = row["SKU"]
            sku = sku[3:].replace(".", "").upper()
            url = f"https://repository.falc.biz/samples/{sku}-5.JPG"
            #caption = get_blip_caption(url)
            caption = get_blip_caption_new(url)
            
        except Exception as e:
            caption = None
    else:
        caption = None
        
    #caption = get_blip_caption(row.get("Image 1", "")) if use_image and row.get("Image 1", "") else None

    prompt = build_unified_prompt(
        row=row,
        col_display_names=col_display_names,
        selected_langs=selected_langs,
        image_caption=caption,
        simili=simili
    )

    # Token estimation (~4 chars per token)
    num_chars = len(prompt)
    token_est = num_chars // 4
    cost_est = round(token_est / 1000 * 0.001, 6)

    if DEBUG:
        st.code(prompt)
        st.markdown(f"📊 **Prompt Length**: {num_chars} caratteri ≈ {token_est} token")
        st.markdown(f"💸 **Costo stimato per riga**: ${cost_est:.6f}")

    return token_est, cost_est, prompt
    
# ---------------------------
# DropBox
# ---------------------------

def get_dropbox_access_token():
    refresh_token = st.secrets["DROPBOX_REFRESH_TOKEN"]
    client_id = st.secrets["DROPBOX_CLIENT_ID"]
    client_secret = st.secrets["DROPBOX_CLIENT_SECRET"]

    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    response = requests.post(
        "https://api.dropbox.com/oauth2/token",
        headers={
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]

def get_dropbox_client():
    access_token = get_dropbox_access_token()
    return dropbox.Dropbox(access_token)

def upload_csv_to_dropbox(dbx, folder_path: str, file_name: str, file_bytes: bytes):
    dbx_path = f"{folder_path}/{file_name}"
    try:
        dbx.files_create_folder_v2(folder_path)
    except dropbox.exceptions.ApiError:
        pass  # cartella già esiste
    try:
        dbx.files_upload(file_bytes, dbx_path, mode=WriteMode("overwrite"))
        
        st.success(f"✅ CSV caricato su Dropbox: {dbx_path}")
    except Exception as e:
        st.error(f"❌ Errore caricando CSV su Dropbox: {e}")

def download_csv_from_dropbox(dbx, folder_path: str, file_name: str) -> io.BytesIO:
    file_path = f"{folder_path}/{file_name}"

    try:
        metadata, res = dbx.files_download(file_path)
        return io.BytesIO(res.content), metadata
    except dropbox.exceptions.ApiError as e:
        # Se l'errore è 'path/not_found' -> file mancante
        if (hasattr(e.error, "is_path") and e.error.is_path() 
                and e.error.get_path().is_not_found()):
            return None, None
        else:
            # altri errori (permessi, connessione, ecc.)
            st.error(f"Errore scaricando da Dropbox: {e}")
            return None, None
            
def format_dropbox_date(dt):
    if dt is None:
        return "Data non disponibile"

    # Dropbox restituisce sempre datetime tz-aware in UTC, ma nel dubbio gestiamo anche i naïve
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    # Convertiamo in fuso orario italiano
    dt_italy = dt.astimezone(ZoneInfo("Europe/Rome"))

    # Data odierna in Italia
    oggi = datetime.now(ZoneInfo("Europe/Rome")).date()

    mesi_it = [
        "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
        "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
    ]

    if dt_italy.date() == oggi:
        return f"Oggi alle {dt_italy.strftime('%H:%M')}"
    else:
        mese = mesi_it[dt_italy.month - 1]
        return f"{dt_italy.day:02d} {mese} {dt_italy.year} - {dt_italy.strftime('%H:%M')}"
    
# ---------------------------
# Funzioni varie
# ---------------------------
def read_csv_auto_encoding(uploaded_file, separatore=None):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    uploaded_file.seek(0)  # Rewind after read
    if separatore:
        return pd.read_csv(uploaded_file, sep=separatore, encoding=encoding, dtype=str)
    else:
        return pd.read_csv(uploaded_file, encoding=encoding, dtype=str)

def not_in_array(array, list):
    missing = not all(col in array for col in list)
    return missing
    
def genera_pdf(df_disp, **param):
    # --- Parametri ---
    truncate_map = param.get("truncate_map", None)  # se None = niente troncamento

    # --- Copia DF per non modificare l'originale ---
    df_proc = df_disp.copy()

    # --- Applico troncamento solo se truncate_map è definito ---
    if truncate_map:
        for col, max_len in truncate_map.items():
            if col in df_proc.columns:
                df_proc[col] = df_proc[col].astype(str).apply(
                    lambda x: x if len(x) <= max_len else x[:max_len-3]
                )

    # --- Altri parametri ---
    font_size = param.get("font_size", 12)
    header_bg_color = param.get("header_bg_color", colors.grey)
    header_text_color = param.get("header_text_color", colors.whitesmoke)
    row_bg_color = param.get("row_bg_color", colors.beige)
    header_align = param.get("header_align", "CENTER")
    text_align = param.get("text_align", "CENTER")
    valign = param.get("valign", "MIDDLE")
    margins = param.get("margins", (20, 20, 30, 20))
    row_height_factor = param.get("row_height_factor", 2.2)
    repeat_header = param.get("repeat_header", True)
    col_widths = param.get("col_widths", {})
    align_map = param.get("align_map", {})

    row_height_default = font_size * row_height_factor
    row_heights = [row_height_default] * (len(df_proc) + 1)

    # --- PDF ---
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margins[0],
        rightMargin=margins[1],
        topMargin=margins[2],
        bottomMargin=margins[3],
    )

    data = [list(df_proc.columns)] + df_proc.values.tolist()
    table = Table(
        data,
        repeatRows=1 if repeat_header else 0,
        hAlign='CENTER',
        rowHeights=row_heights
    )

    if col_widths:
        table._argW = [col_widths.get(col, 60) for col in df_proc.columns]

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), header_text_color),
        ("ALIGN", (0, 0), (-1, 0), header_align),
        ("ALIGN", (0, 1), (-1, -1), text_align),
        ("VALIGN", (0, 0), (-1, -1), valign),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("BACKGROUND", (0, 1), (-1, -1), row_bg_color),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ])

    for col_name, align in align_map.items():
        if col_name in df_proc.columns:
            idx = df_proc.columns.get_loc(col_name)
            style.add("ALIGN", (idx, 0), (idx, -1), align)

    table.setStyle(style)

    elements = [table]
    doc.build(elements)
    buffer.seek(0)
    return buffer

def process_csv_and_update(sheet, uploaded_file, batch_size=100):
    st.text("1️⃣ Leggo CSV...")
    df = read_csv_auto_encoding(uploaded_file)

    expected_cols = [
        "Anno","Stag.","Clz.","Descr.","Serie","Descriz1","Annullato",
        "Campionato","Cat","Cod","Descr2","Var.","DescrizVar","Col.",
        "DescrCol","TAGLIA","QUANTIA","Vuota","DATA_CREAZIONE","N=NOOS"
    ]

    if df.shape[1] != len(expected_cols):
        st.error(f"⚠️ CSV ha {df.shape[1]} colonne invece di {len(expected_cols)}. Controlla separatore o formato!")
        st.dataframe(df.head())
        return 0, 0

    df.columns = expected_cols
    df["SKU"] = df["Cod"].astype(str) + df["Var."].astype(str) + df["Col."].astype(str)

    # Porta SKU come prima colonna
    cols = ["SKU"] + [c for c in df.columns if c != "SKU"]
    df = df[cols]

    st.text("2️⃣ Carico dati esistenti dal foglio...")
    existing_values = sheet.get("A:U")
    if not existing_values:
        header = df.columns.tolist()
        existing_df = pd.DataFrame(columns=header)
    else:
        header = existing_values[0]
        data = existing_values[1:]
        existing_df = pd.DataFrame(data, columns=header)

    existing_df = existing_df.fillna("").astype(str)
    existing_dict = {row["SKU"]: row for _, row in existing_df.iterrows()}

    st.text("3️⃣ Identifico nuove righe e aggiornamenti...")
    new_rows = []
    updates = []

    total = len(df)
    progress = st.progress(0)

    for i, row in df.iterrows():
        sku = row["SKU"]
        new_year_stage = (int(row["Anno"]), int(row["Stag."]))
        single_row = ["" if pd.isna(x) else str(x) for x in row]

        if sku not in existing_dict:
            new_rows.append(single_row)
        else:
            existing_row = existing_dict[sku]
            existing_year_stage = (int(existing_row["Anno"]), int(existing_row["Stag."]))
            if new_year_stage[0] > existing_year_stage[0] or \
               (new_year_stage[0] == existing_year_stage[0] and new_year_stage[1] > existing_year_stage[1]):
                idx = existing_df.index[existing_df["SKU"] == sku][0] + 2  # +2 per header e base 1
                updates.append((idx, single_row))

        if i % 50 == 0:
            progress.progress(i / total)
    progress.progress(1.0)

    st.text(f"✅ Nuove righe da aggiungere: {len(new_rows)}")
    st.text(f"✅ Aggiornamenti da effettuare: {len(updates)}")

    # =======================
    # Aggiornamenti batch
    # =======================
    st.text("4️⃣ Aggiorno righe esistenti in batch...")
    for start in range(0, len(updates), batch_size):
        batch = updates[start:start+batch_size]
        ranges = [f"A{idx}:U{idx}" for idx, _ in batch]
        values = [row for _, row in batch]
        body = [{"range": r, "values": [v]} for r, v in zip(ranges, values)]
        if body:
            sheet.batch_update(body)

    # =======================
    # Aggiunta nuove righe
    # =======================
    st.text("5️⃣ Aggiungo nuove righe in fondo...")
    if new_rows:
        # pulizia valori
        new_rows_clean = [[str(cell) if cell is not None else "" for cell in row] for row in new_rows]
    
        start_row = len(existing_df) + 2  # +2 per header e base 1
        end_row = start_row + len(new_rows_clean) - 1
        missing_rows = end_row - sheet.row_count
        if missing_rows > 0:
            sheet.add_rows(missing_rows)
        cell_range = f"A{start_row}:U{end_row}"
    
        sheet.update(cell_range, new_rows_clean, value_input_option="RAW")

    st.text("✅ Operazione completata!")
    return len(new_rows), len(updates)

# Traduttore singolo (creato una sola volta per efficienza)
TRANSLATION_DB_FILE = "translations_db.json"

def load_translation_db():
    """Carica il file JSON delle traduzioni"""
    if os.path.exists(TRANSLATION_DB_FILE):
        with open(TRANSLATION_DB_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ File DB traduzioni danneggiato, lo ricreo.")
                return []
    return []

def save_translation_db(db):
    """Salva il DB nel file JSON"""
    with open(TRANSLATION_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def find_translation(db, text_it, target_lang):
    """Cerca una traduzione esistente nel DB"""
    for entry in db:
        if entry.get("it", "").strip().lower() == text_it.strip().lower():
            return entry.get(target_lang)
    return None

def add_translation(db, text_it, lang, translated_text):
    """Aggiunge o aggiorna una traduzione nel DB"""
    for entry in db:
        if entry.get("it", "").strip().lower() == text_it.strip().lower():
            entry[lang] = translated_text
            break
    else:
        db.append({"it": text_it, lang: translated_text})
    save_translation_db(db)

def create_translator(source, target):
    return GoogleTranslator(source=source, target=target)

def safe_translate(text, translator):
    """Traduci testo con gestione errori e cache su file"""
    time.sleep(0.1)
    try:
        if not text or str(text).strip() == "":
            return ""

        text_it = str(text).strip()
        target_lang = translator.target
        db = load_translation_db()

        # 1️⃣ Controlla se esiste già nel DB
        cached = find_translation(db, text_it, target_lang)
        if cached:
            return cached

        # 2️⃣ Se non esiste → traduci e salva nel DB
        translated = translator.translate(text_it)
        add_translation(db, text_it, target_lang, translated)
        return translated

    except Exception as e:
        print(f"❌ Errore durante la traduzione: {e}")
        return str(text)  # fallback

def translate_column_parallel(col_values, source, target, max_workers=5):
    """Traduci una colonna mantenendo l'ordine originale"""
    translator = create_translator(source, target)
    results = [None] * len(col_values)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(safe_translate, text, translator): i for i, text in enumerate(col_values)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Errore riga {idx}: {e}")
                results[idx] = str(col_values[idx])

    return results
    
# --- Funzione per generare PDF ---
def genera_pdf_aggrid(df_table, file_path="giac_corridoio.pdf"):
    doc = SimpleDocTemplate(file_path, pagesize=landscape(A4))
    elements = []

    # --- Estrazione marchi ---
    brands = [c.replace("_VECCHIO", "") for c in df_table.columns if "_VECCHIO" in c]

    # --- Header multi-riga ---
    header_row1 = ["CORR"] + [brand for brand in brands for _ in range(2)]
    header_row2 = [""] + ["VECCHIO" if i % 2 == 0 else "NUOVO" for i in range(len(brands)*2)]
    data = [header_row1, header_row2]

    # --- Righe dati ---
    for _, row in df_table.iterrows():
        row_data = [int(row.get("CORR", 0))]
        for brand in brands:
            row_data.append(int(row.get(f"{brand}_VECCHIO", 0)))
            row_data.append(int(row.get(f"{brand}_NUOVO", 0)))
        data.append(row_data)

    # --- Larghezza colonne ---
    col_widths = [40] + [60] * len(brands) * 2
    t = Table(data, colWidths=col_widths)

    # --- Stile tabella ---
    style = TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BACKGROUND', (0,1), (-1,1), colors.whitesmoke),
        ('SPAN', (0,0), (0,1)),
    ])

    # --- SPAN per i brand ---
    col_start = 1
    for _ in brands:
        style.add('SPAN', (col_start,0), (col_start+1,0))
        col_start += 2

    # --- Colori alternati VECCHIO/NUOVO ---
    for i in range(len(brands)):
        col_vecchio = 1 + i*2
        col_nuovo = col_vecchio + 1
        style.add('BACKGROUND', (col_vecchio,2), (col_vecchio,-1), colors.beige)
        style.add('BACKGROUND', (col_nuovo,2), (col_nuovo,-1), colors.lavender)

    t.setStyle(style)
    elements.append(t)
    doc.build(elements)

    return open(file_path, "rb").read()
    
# ---------------------------
# Funzioni gestione foto
# ---------------------------
def genera_lista_sku(sheet_id: str, tab_names: list[str]):
    sheet_lista = get_sheet(sheet_id, "LISTA")
    sku_dict = {}

    for tab in tab_names:
        all_values = get_sheet(sheet_id, tab).get_all_values()
        if len(all_values) <= 2:
            continue

        headers = all_values[1]
        data = all_values[2:]
        df = pd.DataFrame(data)

        for _, row in df.iterrows():
            codice = str(row.get(df.columns[7], "")).zfill(7)
            variante = str(row.get(df.columns[8], "")).zfill(2)
            colore = str(row.get(df.columns[9], "")).zfill(4)

            if codice and variante and colore:
                sku = f"{codice}{variante}{colore}"
                sku_dict.setdefault(sku, set()).add(tab)

    new_rows = []
    for sku in sorted(sku_dict.keys()):
        provenienza = ", ".join(sorted(sku_dict[sku]))
        new_rows.append([sku, provenienza])

    # Scrivi solo colonne A e B da riga 3
    range_update = f"A3:B{len(new_rows)+2}"
    sheet_lista.update(range_update, new_rows, value_input_option="RAW")

def aggiungi_sku(sheet_id: str, sku: str):
    sheet_lista = get_sheet(sheet_id, "LISTA")
    lastRow = sheet_lista.row_count
    row = [sku,
           st.session_state.user["username"],
           f"=SE.ERRORE(CERCA.VERT($A{lastRow+1};ANAGRAFICA!A:C;2;0)&\"/\"&CERCA.VERT($A{lastRow+1};ANAGRAFICA!A:C;3;0);)",
           f"=SE.ERRORE(CERCA.VERT($A{lastRow+1};ANAGRAFICA!A:E;5;0);)",
           f"=SE($C$1=\"X\";CERCA.VERT($A{lastRow+1};Foglio22!$A:$L;12;0);SE.ERRORE(CERCA.VERT($A{lastRow+1};ANAGRAFICA!A:I;9;0);))",
           f"=SE.ERRORE(STRINGA.ESTRAI($A{lastRow+1};1;7);)",
           f"=SE.ERRORE(STRINGA.ESTRAI($A{lastRow+1};8;2);)",
           f"=SE.ERRORE(STRINGA.ESTRAI($A{lastRow+1};10;4);)",
           f"=SE.ERRORE(CERCA.VERT($A{lastRow+1};ANAGRAFICA!$A:$J;10;0);)",
           f"=IFNA(SE(E(O($K{lastRow+1}=\"True\";$P{lastRow+1}=\"True\");$Q{lastRow+1}=\"False\");SE(MIN(ARRAYFORMULA(SE(GIACENZE!$I:$I=$A{lastRow+1};GIACENZE!$D:$D;\"\")))=0;\"\";SE(MIN(ARRAYFORMULA(SE(GIACENZE!$I:$I=$A{lastRow+1};GIACENZE!$D:$D;\"\")))<$I{lastRow+1};SE(SE.ERRORE(CERCA.VERT(CONCATENA($A{lastRow+1};$I{lastRow+1});GIACENZE!$H:$Q;10;0);0)>0;$I{lastRow+1};MIN(ARRAYFORMULA(SE(GIACENZE!$I:$I=$A{lastRow+1};GIACENZE!$D:$D;\"\"))));MIN(ARRAYFORMULA(SE(GIACENZE!$I:$I=$A{lastRow+1};GIACENZE!$D:$D;\"\")))));\"\");$I{lastRow+1})",
           "'False",
           "",
           f"=SE(E(O($K{lastRow+1}=\"True\";$P{lastRow+1}=\"True\");MIN(ARRAYFORMULA(SE(GIACENZE!$I:$I=$A{lastRow+1};GIACENZE!$D:$D;\"\")))>0);SE($Q{lastRow+1}=\"False\";\"True\";\"False\");SE(O($K{lastRow+1}=\"True\";$P{lastRow+1}=\"True\");SE(SINISTRA($A{lastRow+1};1)=\"6\";SE(SE.ERRORE(CERCA.VERT($A{lastRow+1};GIACENZE!$I:$Q;9;0);-1)>0;\"True\";\"False\");\"False\");\"False\"))",
           f"=SE(E(O($K{lastRow+1}=\"True\";$P{lastRow+1}=\"True\");$Q{lastRow+1}=\"False\";$M{lastRow+1}=\"False\";MIN(ARRAYFORMULA(SE(PIM!$I:$I=$A{lastRow+1};PIM!$D:$D;\"\")))>0);SE(SE.ERRORE(CERCA.VERT(CONCATENA($A{lastRow+1};$I{lastRow+1});PIM!$H:$V;15;0);0)>0;\"True\";\"False\");\"False\")",
           f"=SE(E(O($K{lastRow+1}=\"True\";$P{lastRow+1}=\"True\");$Q{lastRow+1}=\"False\";$M{lastRow+1}=\"False\";MIN(ARRAYFORMULA(SE(PIM!$I:$I=$A{lastRow+1};PIM!$D:$D;\"\")))>0);SE(SE.ERRORE(CERCA.VERT(CONCATENA($A{lastRow+1};$I{lastRow+1});PIM!$H:$Y;18;0);0)>0;\"True\";\"False\");\"False\")",
           "",
           f"=SE(VAL.ERRORE(CONFRONTA($A{lastRow+1};PRELEVATE!$A:$A;0));\"False\";\"True\")",
           f"=SE(O($M{lastRow+1}=\"True\";$N{lastRow+1}=\"True\";$O{lastRow+1}=\"True\");SE(VAL.NON.DISP(CONFRONTA($D{lastRow+1};SPLIT(SETTINGS(\"brandMatias\");\",\");0));SE(VAL.NON.DISP(CONFRONTA($D{lastRow+1};SPLIT(SETTINGS(\"brandMatteo\");\",\");0));\"\";\"MATTEO\");\"MATIAS\");\"\")",
           f"=SE($M{lastRow+1}=\"True\";SE(SINISTRA($A{lastRow+1};1)=\"6\";CERCA.VERT($A{lastRow+1};GIACENZE!$I:$O;4;0);TESTO(CERCA.VERT(CONCATENA($A{lastRow+1};$J{lastRow+1});GIACENZE!$H:$O;5;0);\"000\"));\"\")",
           f"=SE($M{lastRow+1}=\"True\";SE(SINISTRA($A{lastRow+1};1)=\"6\";CERCA.VERT($A{lastRow+1};GIACENZE!$I:$O;5;0);TESTO(CERCA.VERT(CONCATENA($A{lastRow+1};$J{lastRow+1});GIACENZE!$H:$O;6;0);\"000\"));\"\")",
           f"=SE($M{lastRow+1}=\"True\";SE(SINISTRA($A{lastRow+1};1)=\"6\";CERCA.VERT($A{lastRow+1};GIACENZE!$I:$O;6;0);TESTO(CERCA.VERT(CONCATENA($A{lastRow+1};$J{lastRow+1});GIACENZE!$H:$O;7;0);\"0\"));\"\")",
           f"=SE($M{lastRow+1}=\"True\";SE(SINISTRA($A{lastRow+1};1)=\"6\";CERCA.VERT($A{lastRow+1};GIACENZE!$I:$O;7;0);TESTO(CERCA.VERT(CONCATENA($A{lastRow+1};$J{lastRow+1});GIACENZE!$H:$O;8;0);\"0\"));\"\")",
           f"=SE($N{lastRow+1}=\"True\";ArrayFormula(TEXTJOIN(\" - \";1;unique(SE('UBICAZIONI 027'!$A:$A=$A{lastRow+1};'UBICAZIONI 027'!$F:$J;\"\"));\"\"));SE($O{lastRow+1}=\"True\";ArrayFormula(TEXTJOIN(\" - \";1;unique(SE('UBICAZIONI 028'!$A:$A=$A{lastRow+1};'UBICAZIONI 028'!$F:$J;\"\"));\"\"));\"\"))",
           f"=REPO($A{lastRow+1})",
           "."
          ]
    sheet_lista.append_row(row, value_input_option="USER_ENTERED")
    #sheet_lista.append_row([sku, st.session_state.user["username"]])

@st.cache_data(ttl=300)
def carica_lista_foto(sheet_id: str, cache_key: str = "") -> pd.DataFrame:
    try:
        sheet = get_sheet(sheet_id, "LISTA")
        values = sheet.get("A3:Y5000")
        if not values:
            return pd.DataFrame()
        
        # ✅ Definizione corretta: 24 intestazioni per colonne A–V
        headers = ["SKU", "CANALE", "STAGIONE", "COLLEZIONE", "DESCRIZIONE", "COD", "VAR", "COL", "TG CAMP", "TG PIC", "SCATTARE", "CONTROLLO", "DISP", "DISP 027", "DISP 028", "RISCATTARE", "CONSEGNATA", "FOTOGRAFO", "COR", "LAT", "X", "Y", "UBI", "REPO", "END"]
        df = pd.DataFrame(values, columns=headers)
        df = df[df["SKU"].notna() & (df["SKU"].str.strip() != "")]

        # 🧹 Normalizza booleani
        def normalize_bool(col):
            return col.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
        
        df["SCATTARE"] = normalize_bool(df["SCATTARE"])
        df["CONSEGNATA"] = normalize_bool(df["CONSEGNATA"])
        df["RISCATTARE"] = normalize_bool(df["RISCATTARE"])
        df["DISP"] = normalize_bool(df["DISP"])
        df["DISP 027"] = normalize_bool(df["DISP 027"])
        df["DISP 028"] = normalize_bool(df["DISP 028"])

        return df[["SKU", "STAGIONE", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE", "RISCATTARE", "CONSEGNATA", "DISP", "DISP 027", "DISP 028", "COD", "VAR", "COL", "TG PIC", "TG CAMP", "FOTOGRAFO", "COR", "LAT", "X", "Y", "UBI"]]
    except Exception as e:
        st.error(f"Errore durante il caricamento: {str(e)}")
        return pd.DataFrame()

# ---------------------------
# Funzioni Dashboard
# ---------------------------
# Funzione per estrarre i dati da una pagina PDF
def extract_data_from_page(page_text):
    data = {}
    
    # Marketplace
    # Ricerca separata per essere più robusti alle variazioni di layout
    marketplace_match = re.search(r"Taglia(.*)Marketplace:", page_text, re.IGNORECASE)
    mp = None
    if marketplace_match:
        if "Naturino" in marketplace_match.group(1).strip():
            mp = "Naturino"
        elif "Candice" in marketplace_match.group(1).strip():
            mp = "Candice Cooper"
        elif "Flowermountain" in marketplace_match.group(1).strip():
            mp = "Flower Mountain"
        elif "Voile" in marketplace_match.group(1).strip():
            mp = "Voile Blanche"
        elif "Falcotto" in marketplace_match.group(1).strip():
            mp = "Falcotto"
        elif "falcotto" in marketplace_match.group(1).strip():
            mp = "Falcotto"
        elif "W6YZ" in marketplace_match.group(1).strip():
            mp = "W6YZ"
        elif "zalando" in marketplace_match.group(1).strip():
            mp = "Zalando"
        elif "amazon" in marketplace_match.group(1).strip():
            mp = "Amazon"
        elif "la_redoute" in marketplace_match.group(1).strip():
            mp = "LaRedoute"
        elif "privalia" in marketplace_match.group(1).strip():
            mp = "Privalia"
        elif "venteprivee" in marketplace_match.group(1).strip():
            mp = "VentePrivee"
        elif "sarenza" in marketplace_match.group(1).strip():
            mp = "Sarenza"
        elif "vertbaudet" in marketplace_match.group(1).strip():
            mp = "Vertbaudet"
        elif "miinto" in marketplace_match.group(1).strip():
            mp = "Miinto"
        else:
            mp = marketplace_match.group(1).strip()
            
        data['Marketplace'] = mp
    else:
        data['Marketplace'] = "N/A"

    # Numero Ordine
    #order_match = re.search(r"Marketplace order\s*([a-zA-Z0-9-]+)", page_text, re.IGNORECASE)
    order_match = re.search(r"Marketplace order\s*(.*)\sUbicazioneData vendita", page_text, re.IGNORECASE)
    
    if order_match:
        data['Numero Ordine'] = order_match.group(1).strip()
    else:
        data['Numero Ordine'] = "N/A"

    # Data
    date_match = re.search(r"Data vendita:\s*(\d{2}/\d{2}/\d{4})", page_text)
    if date_match:
        data['Data'] = date_match.group(1)
    else:
        data['Data'] = "N/A"

    # Nazione
    #country_match = re.search(r"\n[0-9]{3}.*([A-Z]{2})\s*$", page_text.strip(), re.MULTILINE)

    nazione = None
    marketplace = marketplace_match.group(1).strip()
    order = order_match.group(1).strip()
    if order.startswith("101") or order.startswith("CC16") or order.startswith("DE_") or "DE" in order or "de" in order:
        nazione = "DE"
    elif order.startswith("103") or order.startswith("CC15") or order.startswith("FR_") or "FR" in order or "fr" in order:
        nazione = "FR"
    elif order.startswith("104") or order.startswith("CC101") or order.startswith("IT_") or "IT" in order or "it" in order:
        nazione = "IT"
    elif order.startswith("CC11"):
        nazione = "WE"
    elif order.startswith("ES_") or "ES" in order:
        nazione = "ES"
    elif order.startswith("BE_"):
        nazione = "BE"
    elif order.startswith("NL_"):
        nazione = "NL"
    elif order.startswith("SW_"):
        nazione = "SE"
    elif "WE" in order:
        nazione = "WE"
    elif "GB" in order:
        nazione = "GB"
    elif "US" in order:
        nazione = "US"
    else:
        if "DE" in marketplace or "de" in marketplace:
            nazione = "DE"
        elif "FR" in marketplace or "fr" in marketplace or "sarenza" in marketplace:
            nazione = "FR"
        elif "IT" in marketplace or "it" in marketplace:
            nazione = "IT"
        elif "ES" in marketplace or "es" in marketplace:
            nazione = "ES"
        else:
            nazione = "N/A"
    
    data['Nazione'] = nazione
    
    #if country_match:
    #    data['Nazione'] = country_match.group(1).strip()
    #else:
    #    data['Nazione'] = "N/A"
    
    # Articoli, quantità e taglia
    items = []
    items_section = re.search(r"Articolo\s+Quantità\s+Descrizione articolo\s+Taglia(.*?)ISTRUZIONI IMBALLO", page_text, re.DOTALL)
    if not items_section:
        items_section = re.search(r"Articolo\s+Taglia\s+Quantità\s+Descrizione articolo(.*?)ISTRUZIONI IMBALLO", page_text, re.DOTALL)
    
    if items_section:
        items_text = items_section.group(1).strip()
        item_lines = items_text.split('\n')
        for line in item_lines:
            if line.strip():
                # Adjusted regex to handle different formats and include size
                # Case 1: Quantity Code Size Description
                #item_match = re.search(r"^\s*(\d+)\s+([^\s]+)\s+(\d+)\s+(.*)", line)
                item_match = re.search(r"^\s*(\d+)\s+001([a-zA-Z0-9]{7})\.([a-zA-Z0-9]{2})\.([a-zA-Z0-9]{4})\s+(\d+)\s+(.*)", line)
                if item_match:
                    item_data = {
                        'quantita': item_match.group(1).strip(),
                        'codice': item_match.group(2).strip() + item_match.group(3).strip() + item_match.group(4).strip(),
                        'taglia': item_match.group(5).strip(),
                        'descrizione': item_match.group(6).strip()
                    }
                    items.append(item_data)
                else:
                    # Case 2: Code Size Quantity Description
                    #item_match = re.search(r"^\s*([^\s]+)\s+(\d+)\s+(\d+)\s+(.*)", line)
                    item_match = re.search(r"^\s*001(.*)\.(.*)\.(.*){4}\s+(\d+)\s+(\d+)\s+(.*)", line)
                    if item_match:
                        item_data = {
                            'codice': item_match.group(1).strip() + item_match.group(2).strip() + item_match.group(3).strip(),
                            'taglia': item_match.group(4).strip(),
                            'quantita': item_match.group(5).strip(),
                            'descrizione': item_match.group(6).strip()
                        }
                        items.append(item_data)
                        
    data['Articoli'] = items
    
    return data

def get_country_from_address(address):
    """
    Tenta di estrarre il nome del paese da una stringa di indirizzo utilizzando il geocoding.

    Args:
        address (str): L'indirizzo da analizzare.

    Returns:
        str: Il nome del paese o "N/A" se non è stato trovato.
    """
    try:
        # Crea un geocoder con un timeout per evitare che il programma si blocchi
        geolocator = Nominatim(user_agent="my-app", timeout=5)
        
        # Tenta di geocodificare l'indirizzo
        location = geolocator.geocode(address)
        
        if location and location.address:
            # L'indirizzo geocodificato restituisce una stringa strutturata.
            # Esempio: 'Via Roma 1, 00100 Roma RM, Italia'
            # Usiamo una regex per trovare il nome del paese alla fine della stringa.
            country_match = re.search(r"([^,]+)\s*$", location.address)
            if country_match:
                return country_match.group(1)
        
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        print(f"Errore di geocoding: {e}")
        
    return "N/A"

    
# ---------------------------
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Gestione ECOM", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 300px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 📊 Google Sheets
# ---------------------------
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
)
gsheet_client = gspread.authorize(credentials)
drive_service = build('drive', 'v3', credentials=credentials)

def get_sheet(sheet_id, tab):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    worksheets = spreadsheet.worksheets()
    
    # Confronto case-insensitive per maggiore robustezza
    for ws in worksheets:
        if ws.title.strip().lower() == tab.strip().lower():
            return ws

    # Se non trovato, lo crea
    return spreadsheet.add_worksheet(title=tab, rows="10000", cols="50")

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    sheet.append_rows(values, value_input_option="RAW")  # ✅ chiamata unica

def append_logs(sheet_id, logs_data):
    sheet = get_sheet(sheet_id, "logs")
    values = [list(log.values()) for log in logs]
    sheet.append_rows(values, value_input_option="RAW")
    
def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")


# 📁 Caricamento dati
# Sidebar: menu
with st.sidebar:
    DEBUG = st.checkbox("🪛 Debug")
    # Togliere per riattivare password e nome
    #st.session_state["logged_as"] = "GUEST"
    if DEBUG:
        st.session_state.user = {
            "data": "data",
            "email": "test@test.it",
            "nome": "GUEST",
            "cognome": "Test2",
            "username": "Username",
            "role": "admin"
        }

    if "user" not in st.session_state or st.session_state.user is None:
        page = "Home"
        st.markdown("## 🔑 Login")
        with st.form("login_user"):
            email = st.text_input("Username")
            password = st.text_input("Password", type="password")

            login_button = st.form_submit_button("Accedi")
            
        if login_button:
            if login(email, password):
                st.rerun()  # ricarica subito la pagina senza messaggio
    else:
        user = st.session_state.user
        st.write(f"Accesso eseguito come: {user["nome"]}")

        menu_item_list = [{"name":"Home", "icon":"house", "role":["guest","logistica","customer care","admin"]},
                          {"name":"Catalogo", "icon":"list", "role":["logistica","customer care","admin"]},
                          {"name":"Ordini", "icon":"truck", "role":["logistica","customer care","admin"]},
                          {"name":"Descrizioni", "icon":"list", "role":["customer care","admin"]},
                          {"name":"Foto", "icon":"camera", "role":["logistica","customer care","admin"]},
                          {"name":"Giacenze", "icon":"box", "role":["logistica","customer care","admin"]},
                          {"name":"Ferie", "icon":"palm", "role":["admin"]},
                          {"name":"Admin", "icon":"gear", "role":["admin"]},
                          {"name":"Test", "icon":"gear", "role":["admin"]},
                          {"name":"Logout", "icon":"key", "role":["guest","logistica","customer care","admin"]}
                         ]
        
        submenu_item_list = [{"main":"Catalogo", "name":"Trova articolo", "icon":"search", "role":["logistica","customer care","admin"]},
                             {"main":"Catalogo", "name":"Aggiungi ordini stagione", "icon":"plus", "role":["logistica","customer care","admin"]},
                             {"main":"Ordini", "name":"Dashboard", "icon":"bar-chart", "role":["admin"]},
                             {"main":"Ordini", "name":"Importa", "icon":"plus", "role":["admin"]},
                             {"main":"Foto", "name":"Gestione", "icon":"gear", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Riscatta SKU", "icon":"repeat", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Aggiungi SKUs", "icon":"plus", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Storico", "icon":"book", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Foto", "name":"Aggiungi prelevate", "icon":"hand-index", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Importa", "icon":"download", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Per corridoio", "icon":"1-circle", "role":["guest","logistica","admin"]},
                             {"main":"Giacenze", "name":"Per corridoio/marchio", "icon":"2-circle", "role":["guest","logistica","admin"]},
                             {"main":"Giacenze", "name":"Aggiorna anagrafica", "icon":"refresh", "role":["guest","logistica","customer care","admin"]},
                             {"main":"Giacenze", "name":"Old import", "icon":"download", "role":["admin"]},
                             {"main":"Ferie", "name":"Report", "icon":"list", "role":["admin"]},
                             {"main":"Ferie", "name":"Aggiungi ferie", "icon":"plus", "role":["admin"]},
                             {"main":"Admin", "name":"Aggiungi utente", "icon":"plus", "role":["admin"]}
                            ]
        
        menu_items = []
        icon_items = []
        for item in menu_item_list:
            if user["role"] in item["role"]:
                menu_items.append(item["name"])
                icon_items.append(item["icon"])
        
        
        st.markdown("## 📋 Menu")
        # --- Menu principale verticale ---
        main_page = option_menu(
            menu_title=None,
            options=menu_items,
            icons=icon_items,
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "2px",
                    "padding": "5px 10px",
                    "border-radius": "5px",
                    "--hover-color": "#e0e0e0",
                },
                "nav-link-selected": {
                    "background-color": "#4CAF50",
                    "color": "white",
                    "border-radius": "5px",
                },
            },
        )

        # Rimuovo icone/emoji per gestire page name
        main_page_name = main_page

        page = main_page_name  # default

        submenu_items = []
        submenu_icons = []
        for item in submenu_item_list:
            if main_page == item["main"] and user["role"] in item["role"]:
                submenu_items.append(item["name"])
                submenu_icons.append(item["icon"])
                
        if submenu_items:
            sub_page = option_menu(
                menu_title=None,
                options=submenu_items,
                icons=submenu_icons,
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "0!important", "background-color": "#f0f0f0"},
                    "nav-link": {
                        "font-size": "15px",
                        "text-align": "left",
                        "margin": "2px",
                        "padding": "5px 15px",
                        "border-radius": "5px",
                        "--hover-color": "#e0e0e0",
                    },
                    "nav-link-selected": {
                        "background-color": "#4CAF50",
                        "color": "white",
                        "border-radius": "5px",
                    },
                },
            )
            page = f"{main_page_name} - {sub_page}"



# ---------------------------
# 🏠 HOME
# ---------------------------
if page == "Home":
    prompt = st.text_input("Inserisci il tuo prompt:")
    if st.button("Invia"):
        result = test_mistral(prompt)
        st.write("Risposta:", result["choices"][0]["message"]["content"])
    viste.homepage.view()

# ---------------------------
# 🏠 LOGIN
# ---------------------------
if page == "🔑 Login":
    st.subheader("📌 Login")
    st.markdown("""
    Inserisci il nome per poter eseguire azioni.
    """)

    if not st.session_state.get("password_ok"):
        password = st.text_input("Password", key="password_input", type="password", on_change=None)
        if password:
            if password != "Supr3m4@00":
                st.warning("Password errata!")
            else:
                st.session_state["password_ok"] = True
                st.rerun()
    else:
        st.success("Password corretta")
        login = st.text_input("Nome", key="login_input")
        if login:
            login_as(login)

        
# ---------------------------
# 📝 GENERAZIONE DESCRIZIONI
# ---------------------------
elif page == "Descrizioni":
    st.header("📥 Caricamento CSV dei prodotti")
    
    uploaded = st.file_uploader("Carica un file CSV", type="csv")
    
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
    
            def_column = ["Saison", "Silouhette",
                          "sole_material_zalando",
                          "shoe_fastener_zalando",
                          "upper_material_zalando",
                          "futter_zalando",
                          "Sp.feature"
                         ]
            trans_def_colum = {"Saison": "Stagione",
                               "Silouhette": "Tipo di calzatura",
                               "sole_material_zalando": "Soletta interna",
                               "shoe_fastener_zalando": "Chiusura",
                               "upper_material_zalando": "Tomaia",
                               "futter_zalando": "Fodera interna",
                               "Sp.feature": "Caratteristica"
                              }
    
            missing = not_in_array(df_input.columns, def_column)
            if missing:
                def_column = []
                
            st.session_state.selected_cols = st.multiselect("Colonne da includere nel prompt", options=available_cols, default=def_column)
    
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
                            #f"Etichetta: {col}", value=st.session_state.col_display_names[col], key=f"label_{col}"
                            f"Etichetta: {col}", value=trans_def_colum[col], key=f"label_{col}"
                        )
    
        # 🌍 Lingue e parametri
        with st.expander("🌍 Selezione Lingue & Parametri"):
            settings_col1, settings_col2, settings_col3 = st.columns(3)
            with settings_col1:
                marchio = st.radio(
                    "Seleziona il marchio",
                    ["NAT", "FAL", "VB", "FM", "WZ", "CC"],
                    horizontal = False
                )
                use_simili = st.checkbox("Usa descrizioni simili (RAG)", value=True)
                k_simili = 2 if use_simili else 0
                
                use_image = st.checkbox("Usa immagine per descrizioni accurate", value=True)

                use_model = st.radio("Seleziona modello GPT", ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4.1-nano", "gpt-5-nano", "mistral-medium", "deepseek-chimera"], index=0, horizontal = True)
    
            with settings_col2:
                selected_labels = st.multiselect(
                    "Lingue di output",
                    options=list(LANG_LABELS.keys()),
                    default=["Italiano", "Inglese", "Francese", "Tedesco", "Spagnolo"]
                )
                selected_langs = [LANG_LABELS[label] for label in selected_labels]
                
                selected_tones = st.multiselect(
                    "Tono desiderato",
                    ["professionale", "amichevole", "accattivante", "descrittivo", "tecnico", "ironico", "minimal", "user friendly", "SEO-friendly", "SEO-optimized"],
                    default=["professionale", "user friendly", "SEO-friendly", "SEO-optimized"]
                )
    
            with settings_col3:
                desc_lunga_length = st.selectbox("Lunghezza descrizione lunga", ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"], index=5)
                desc_breve_length = st.selectbox("Lunghezza descrizione breve", ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"], index=1)
    
        # 💵 Stima costi
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
                st.info(f"""
                📊 Token totali: ~{token_est}
                💸 Costo stimato: ${cost_est:.6f}
                """)
    
        # 🪄 Generazione descrizioni
        if not check_openai_key():
            st.error("❌ La chiave OpenAI non è valida o mancante. Inserisci una chiave valida prima di generare descrizioni.")
        else:
            if st.button("🚀 Genera Descrizioni"):
                st.session_state["generate"] = True
            
            if st.session_state.get("generate"):
                logs = []
                try:
                    with st.spinner("📚 Carico storico e indice FAISS..."):
                        tab_storico = f"STORICO_{marchio}"
                        data_sheet = get_sheet(desc_sheet_id, tab_storico)
                        df_storico = pd.DataFrame(data_sheet.get_all_records()).tail(500)
            
                        if "faiss_index" not in st.session_state:
                            index, index_df = build_faiss_index(df_storico, st.session_state.col_weights)
                            st.session_state["faiss_index"] = (index, index_df)
                        else:
                            index, index_df = st.session_state["faiss_index"]
            
                    # ✅ Recupera descrizioni già esistenti su GSheet
                    st.info("🔄 Verifico se alcune righe sono già state generate...")
                    existing_data = {}
                    already_generated = {lang: [] for lang in selected_langs}
                    rows_to_generate = []
            
                    for lang in selected_langs:
                        try:
                            tab_df = pd.DataFrame(get_sheet(desc_sheet_id, lang).get_all_records())
                            tab_df = tab_df[["SKU", "Description", "Description2"]].dropna(subset=["SKU"])
                            tab_df["SKU"] = tab_df["SKU"].astype(str)
                            existing_data[lang] = tab_df.set_index("SKU")
                        except:
                            existing_data[lang] = pd.DataFrame(columns=["Description", "Description2"])
            
                    for i, row in df_input.iterrows():
                        sku = str(row.get("SKU", "")).strip()
                        if not sku:
                            rows_to_generate.append(i)
                            continue
            
                        all_present = True
                        for lang in selected_langs:
                            df_lang = existing_data.get(lang)
                            if df_lang is None or sku not in df_lang.index:
                                all_present = False
                                break
                            desc = df_lang.loc[sku]
                            if not desc["Description"] or not desc["Description2"]:
                                all_present = False
                                break
            
                        if all_present:
                            for lang in selected_langs:
                                desc = existing_data[lang].loc[sku]
                                output_row = row.to_dict()
                                output_row["Description"] = desc["Description"]
                                output_row["Description2"] = desc["Description2"]
                                already_generated[lang].append(output_row)
                        else:
                            rows_to_generate.append(i)
            
                    df_input_to_generate = df_input.iloc[rows_to_generate]
            
                    # Costruzione dei prompt
                    all_prompts = []
                    with st.spinner("✍️ Costruisco i prompt..."):
                        for _, row in df_input_to_generate.iterrows():
                            simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights) if k_simili > 0 else pd.DataFrame([])
                            if use_image:
                                try:
                                    sku = row.get("SKU", "")
                                    sku = sku[3:].replace(".", "").upper()
                                    url = f"https://repository.falc.biz/samples/{sku}-5.JPG"
                                    #caption = get_blip_caption(url)
                                    caption = get_blip_caption_new(url)
                                except Exception as e:
                                    caption = None
                            else:
                                caption = None
                            
                            prompt = build_unified_prompt(row, st.session_state.col_display_names, selected_langs, image_caption=caption, simili=simili)
                            all_prompts.append(prompt)
            
                    with st.spinner("🚀 Generazione asincrona in corso..."):
                        if use_model == "mistral-medium":
                            results = asyncio.run(generate_all_prompts_mistral(all_prompts, use_model))
                        elif use_model == "deepseek-chimera":
                            results = asyncio.run(generate_all_prompts_deepseek(all_prompts, use_model))
                        else:
                            results = asyncio.run(generate_all_prompts(all_prompts, use_model))
                    
                    # Parsing risultati
                    all_outputs = already_generated.copy()
                    
                    for i, (_, row) in enumerate(df_input_to_generate.iterrows()):
                        result = results.get(i, {})
                        if "error" in result:
                            logs.append({
                                "utente": st.session_state.user["username"],
                                "sku": row.get("SKU", ""),
                                "status": f"Errore: {result['error']}",
                                "prompt": all_prompts[i],
                                "output": "",
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            continue

                        sku_generate_lista = []
                        for lang in selected_langs:
                            output_row = row.to_dict()
                            lang_data = result.get("result", {}).get(lang.lower(), {})
                            descr_lunga = lang_data.get("desc_lunga", "").strip()
                            descr_breve = lang_data.get("desc_breve", "").strip()
                            output_row["Description"] = descr_lunga
                            output_row["Description2"] = descr_breve
                            all_outputs[lang].append(output_row)
            
                        log_entry = {
                            "utente": st.session_state.user["username"],
                            "sku": row.get("SKU", ""),
                            "status": "OK",
                            "prompt": all_prompts[i],
                            "output": json.dumps(result["result"], ensure_ascii=False),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        if "usage" in result:
                            usage = result["usage"]
                            log_entry.update({
                                "prompt_tokens": usage.get("prompt_tokens", 0),
                                "completion_tokens": usage.get("completion_tokens", 0),
                                "total_tokens": usage.get("total_tokens", 0),
                                "estimated_cost_usd": round(usage.get("total_tokens", 0) / 1000 * 0.001, 6)
                            })
                        logs.append(log_entry)

                    # 🔄 Salvataggio solo dei nuovi risultati
                    with st.spinner("📤 Salvataggio nuovi dati..."):
                        try:
                            for lang in selected_langs:
                                df_out = pd.DataFrame(all_outputs[lang])
                                df_new = df_out[df_out["SKU"].isin(df_input_to_generate["SKU"].astype(str))]
                                if not df_new.empty:
                                    append_to_sheet(desc_sheet_id, lang, df_new)

                            append_logs(desc_sheet_id, logs)
                            #for log in logs:
                            #    append_log(desc_sheet_id, log)
                        except Exception as e:
                            st.warning(f"Errore: {e}")

                    # 📦 ZIP finale
                    with st.spinner("📦 Generazione ZIP..."):
                        mem_zip = BytesIO()
                        with zipfile.ZipFile(mem_zip, "w") as zf:
                            for lang in selected_langs:
                                df_out = pd.DataFrame(all_outputs[lang])
                                df_out["Code langue"] = lang.lower()
                                df_out['Subtitle_trad'] = translate_column_parallel(df_out['Subtitle'].fillna("").tolist(),source='it', target=lang.lower(), max_workers=5)
                                df_out['Subtile2_trad'] = translate_column_parallel(df_out['Subtile2'].fillna("").tolist(),source='it', target=lang.lower(), max_workers=5)
                                
                                df_export = pd.DataFrame({
                                    "SKU": df_out.get("SKU", ""),
                                    "Subtitle": df_out.get("Subtitle_trad", ""),
                                    "Subtile2": df_out.get("Subtile2_trad", ""),
                                    "Descrizione lunga": df_out.get("Description", ""),
                                    "Descrizione breve": df_out.get("Description2", "")
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
                        if desc_sheet_id:
                            tab_storico = f"STORICO_{marchio}"
                            data_sheet = get_sheet(desc_sheet_id, tab_storico)
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
                            #caption = get_blip_caption(image_url) if image_url else None
                            try:
                                sku = row.get("SKU", "")
                                sku = sku[3:].replace(".", "").upper()
                                url = f"https://repository.falc.biz/samples/{sku}-5.JPG"
                                #caption = get_blip_caption(url)
                                caption = get_blip_caption_new(url)
                            except Exception as e:
                                caption = None
                        else:
                            caption = None
                        prompt_preview = build_unified_prompt(test_row, st.session_state.col_display_names, selected_langs, image_caption=caption, simili=simili)
                        st.expander("📄 Prompt generato").code(prompt_preview, language="markdown")
                    except Exception as e:
                        st.error(f"Errore: {str(e)}")
    
            if st.button("🧪 Esegui Benchmark FAISS"):
                with st.spinner("In corso..."):
                    benchmark_faiss(df_input, st.session_state.col_weights)

elif page == "Foto - Gestione":
    st.header("📸 Gestione Foto")
    tab_names = ["ECOM", "ZFS", "AMAZON"]

    selected_ristampe = st.session_state.get("ristampe_selezionate", set())
    
    # 🔽 Caricamento dati con chiave cache dinamica
    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(foto_sheet_id, cache_key=cache_token)
    st.session_state["df_lista_foto"] = df

    # 1️⃣ Genero le liste per i fotografi
    df_disp = df[df["DISP"] == True]
    df_disp_027 = df[df["DISP 027"] == True]
    df_disp_028 = df[df["DISP 028"] == True]
    
    df_disp["X"] = df_disp["X"].astype(int)
    df_disp["Y"] = df_disp["Y"].astype(int)
    
    #df_disp = df_disp[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y","FOTOGRAFO"]]
    df_disp = df_disp.sort_values(by=["COR", "X", "Y", "LAT"])
    df_disp_027 = df_disp_027.sort_values(by=["UBI"])
    df_disp_028 = df_disp_028.sort_values(by=["UBI"])

    df_matias = df_disp[df_disp["FOTOGRAFO"] == "MATIAS"]
    df_matias_027 = df_disp_027[df_disp_027["FOTOGRAFO"] == "MATIAS"]
    df_matias_028 = df_disp_028[df_disp_028["FOTOGRAFO"] == "MATIAS"]
    
    df_matteo = df_disp[df_disp["FOTOGRAFO"] == "MATTEO"]
    df_matteo_027 = df_disp_027[df_disp_027["FOTOGRAFO"] == "MATTEO"]
    df_matteo_028 = df_disp_028[df_disp_028["FOTOGRAFO"] == "MATTEO"]
    

    df_matias = df_matias[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y"]]
    df_matias_027 = df_matias_027[["COD","VAR","COL","TG CAMP","DESCRIZIONE","UBI"]]
    df_matias_028 = df_matias_028[["COD","VAR","COL","TG CAMP","DESCRIZIONE","UBI"]]
    
    df_matteo = df_matteo[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y"]]
    df_matteo_027 = df_matteo_027[["COD","VAR","COL","TG CAMP","DESCRIZIONE","UBI"]]
    df_matteo_028 = df_matteo_028[["COD","VAR","COL","TG CAMP","DESCRIZIONE","UBI"]]

    # 📊 Riepilogo
    total = len(df)
    consegnate = df["CONSEGNATA"].sum()
    da_scattare = df["SCATTARE"].sum() - consegnate
    scattate = total - da_scattare
    #matias = df[df["FOTOGRAFO"] == "MATIAS"].shape[0]
    matias = df_matias.shape[0]
    matias_027 = df_matias_027.shape[0]
    matias_028 = df_matias_028.shape[0]
    
    #matteo = df[df["FOTOGRAFO"] == "MATTEO"].shape[0]
    matteo = df_matteo.shape[0]
    matteo_027 = df_matteo_027.shape[0]
    matteo_028 = df_matteo_028.shape[0]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Avvia workflow"):
            workflow("MarcoRipari", "Gestione-Ecom", "check_photos")
            
        if st.button("📦 Genera lista SKU"):
            try:
                genera_lista_sku(foto_sheet_id, tab_names)
                st.toast("✅ Lista SKU aggiornata!")
            except Exception as e:
                st.error(f"Errore: {str(e)}")
        if st.button("🔄 Refresh"):
            st.session_state["refresh_foto_token"] = str(time.time())
            
        if st.button("Esegui controllo2"):
            url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW_FILENAME}/dispatches"
        
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            }
        
            data = {
                "ref": REF,
                "inputs": {}
            }
        
            response = requests.post(url, headers=headers, json=data)
        
            if response.status_code == 204:
                st.success("✅ Workflow avviato con successo!")
            else:
                st.error(f"❌ Errore: {response.status_code} - {response.text}")
            
    with col2:
        c1, c2, c3, c4, c5 = st.columns([0.35,1,1,1,1])
        c2.metric('📝 Totale SKU', total)
        c3.metric("✅ Già scattate", scattate)
        c4.metric("🚚 Dal fotografo", consegnate)
        c5.metric("📸 Da scattare", da_scattare)
        st.divider()
    

    col_dati1, col_dati2, col_dati3 = st.columns(3)
    with col_dati1:
        t1, t2, t3 = st.columns(3)
        with t2:
            st.subheader("MATIAS")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("ECOM", matias)
            if df_disp.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias),
                    file_name="lista_disp_matias.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="1"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias),
                    file_name="lista_disp_matias.pdf",
                    mime="application/pdf",
                    width="content",
                    key="2"
                )
        with m2:
            st.metric("027", matias_027)
            if df_disp_027.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias_027),
                    file_name="lista_disp_matias_027.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="3"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias_027),
                    file_name="lista_disp_matias_027.pdf",
                    mime="application/pdf",
                    width="content",
                    key="4"
                )
        with m3:
            st.metric("028", matias_028)
            if df_disp_028.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias_028),
                    file_name="lista_disp_matias_028.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="5"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matias_028),
                    file_name="lista_disp_matias_028.pdf",
                    mime="application/pdf",
                    width="content",
                    key="6"
                )

    with col_dati3:
        t4, t5, t6 = st.columns(3)
        with t5:
            st.subheader("MATTEO")
            
        m4, m5, m6 = st.columns(3)
        with m4:
            m4.metric("ECOM", matteo)
            if df_disp.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo),
                    file_name="lista_disp_matteo.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="7"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo),
                    file_name="lista_disp_matteo.pdf",
                    mime="application/pdf",
                    width="content",
                    key="8"
                )

        with m5:
            m5.metric("027", matteo_027)
            if df_disp_027.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo_027),
                    file_name="lista_disp_matteo_027.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="9"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo_027),
                    file_name="lista_disp_matteo_027.pdf",
                    mime="application/pdf",
                    width="content",
                    key="10"
                )
            
        with m6:
            m6.metric("028", matteo_028)
            if df_disp_028.empty:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo_028),
                    file_name="lista_disp_matteo_028.pdf",
                    mime="application/pdf",
                    disabled=True,
                    width="content",
                    key="11"
                )
            else:
                st.download_button(
                    label="📥",
                    data=genera_pdf(df_matteo_028),
                    file_name="lista_disp_matteo_028.pdf",
                    mime="application/pdf",
                    width="content",
                    key="12"
                )
    
   
    # 🔽 Filtro visualizzazione
    filtro_foto = st.selectbox("📌 Filtro foto da fare", ["Tutti", "Solo da scattare", "Solo già scattate", "Solo da riscattare", "Disponibili da prelevare", "Disponibili per Matias", "Disponibili per Matteo"])

    if df.empty:
        st.warning("Nessuna SKU disponibile.")
    else:
        # 🔍 Applica filtro
        if filtro_foto == "Solo da scattare":
            df = df[df["SCATTARE"] == True]
        elif filtro_foto == "Solo già scattate":
            df = df[df["SCATTARE"] == False]
        elif filtro_foto == "Solo da riscattare":
            df = df[df["RISCATTARE"] == True]
        elif filtro_foto == "Disponibili da prelevare":
            df = df[df["DISP"] == True]
        elif filtro_foto == "Disponibili per Matias":
            df = df[df["FOTOGRAFO"] == "MATIAS"]
        elif filtro_foto == "Disponibili per Matteo":
            df = df[df["FOTOGRAFO"] == "MATTEO"]

        # ✅ Visualizzazione con emoji
        def format_checkbox(val):
            if val is True:
                return "✅"
            elif val is False:
                return "❌"
            return "⛔️"

        # ✅ Costruzione vista tabellare con emoji
        df_vista = df.copy()
        df_vista = df_vista[["SKU", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE", "RISCATTARE"]]
        
        df_vista["📷"] = df_vista["SCATTARE"].apply(format_checkbox)
        df_vista["🔁"] = df_vista["RISCATTARE"].apply(format_checkbox)
        
        # Rimuovi le colonne booleane originali
        df_vista = df_vista.drop(columns=["SCATTARE", "RISCATTARE"])
        
        # ✅ Visualizzazione
        st.dataframe(df_vista, use_container_width=True)

elif page == "Foto - Riscatta SKU":
    selected_ristampe = st.session_state.get("ristampe_selezionate", set())

    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(foto_sheet_id, cache_key=cache_token)

    # Foto da riscattare
    st.subheader("🔁 Ristampa foto specifica")
    # ✅ Considera solo SKU che hanno già la foto (SCATTARE == False)
    df_foto_esistenti = df[df["SCATTARE"] == False]

    start_riscattare = len(df[df["RISCATTARE"] == True].index)
    
    if "ristampe_selezionate" not in st.session_state or not st.session_state["ristampe_selezionate"]:
        st.session_state["ristampe_selezionate"] = set(df[df["RISCATTARE"] == True]["SKU"])
    
    selected_ristampe = st.session_state["ristampe_selezionate"]
        
    if st.session_state.get("ristampe_confermate"):
        st.success("✅ Ristampe confermate per le seguenti SKU:")
        for riga in st.session_state["ristampe_confermate"]:
            st.markdown(f"- {riga}")
        time.sleep(5)
        st.session_state["ristampe_confermate"] = False
        st.session_state.ristampa_input = ""
        st.rerun()
    else:
        sku_input = st.text_input("🔍 Inserisci SKU da cercare (solo con foto esistenti)", key="ristampa_input")
        
        if sku_input:
            sku_norm = sku_input.strip().upper()
            match = df_foto_esistenti[df_foto_esistenti["SKU"] == sku_norm]
    
            if match.empty:
                st.warning("❌ SKU non trovata o la foto non esiste ancora.")
            else:
                row = match.iloc[0]
                image_url = f"https://repository.falc.biz/fal001{row['SKU'].lower()}-1.jpg"
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    st.image(image_url, width=100, caption=row["SKU"])
                with cols[1]:
                    st.markdown(f"**{row['DESCRIZIONE']}**")
                    st.markdown(f"*Canale*: {row['CANALE']}  \n*Collezione*: {row['COLLEZIONE']}")
                with cols[2]:
                    if row['SKU'] in selected_ristampe:
                        ristampa_checkbox = st.checkbox("🔁 Ristampa", value=True, key=f"ristampa_{row['SKU']}")
                    else:
                        ristampa_checkbox = st.checkbox("🔁 Ristampa", value=False, key=f"ristampa_{row['SKU']}")
                        
                    if ristampa_checkbox:
                        selected_ristampe.add(row['SKU'])
                    else:
                        selected_ristampe.discard(row['SKU'])

        # Assicurati che lo stato iniziale esista (solo alla prima run)
        if "ristampe_selezionate" not in st.session_state:
            st.session_state["ristampe_selezionate"] = set(df[df["RISCATTARE"] == True]["SKU"].astype(str).tolist())
        
        selected_ristampe = st.session_state["ristampe_selezionate"]
        
        # Qui mostri la lista selezionata (se ce ne sono)
        if selected_ristampe:
            st.markdown(f"📦 SKU selezionate per ristampa: `{', '.join(sorted(selected_ristampe))}`")
        
        # Unico pulsante di conferma (chiave esplicita per evitare collisioni)
        if st.button("✅ Conferma selezione per ristampa", key="conferma_ristampa"):
            try:
                sheet = get_sheet(foto_sheet_id, "LISTA")
                all_rows = sheet.get_all_values()
                data_rows = all_rows[2:]
        
                col_sku = 0
                col_descrizione = 4
                nuovi_valori = []
                sku_descrizioni_confermate = []
        
                for row in data_rows:
                    sku = row[col_sku].strip() if len(row) > col_sku else ""
                    descrizione = row[col_descrizione].strip() if len(row) > col_descrizione else ""
        
                    if sku in selected_ristampe:
                        nuovi_valori.append(["True"])
                        sku_descrizioni_confermate.append(f"{sku} - {descrizione}")
                    else:
                        nuovi_valori.append([""])
        
                range_update = f"P3:P{len(nuovi_valori) + 2}"
                sheet.update(values=nuovi_valori, range_name=range_update)
        
                # 🔄 Ricarico il DataFrame dal Google Sheet (stesso metodo usato sopra)
                df = carica_lista_foto(foto_sheet_id, cache_key=str(time.time()))
                st.session_state["df_lista_foto"] = df
        
                # 🔄 Aggiorno la lista in session_state dai nuovi valori
                st.session_state["ristampe_selezionate"] = set(df[df["RISCATTARE"] == True]["SKU"])
        
                # Salvo anche le confermate
                st.session_state["ristampe_confermate"] = sku_descrizioni_confermate
        
                #st.success("✅ Ristampe aggiornate correttamente!")
                st.rerun()
        
            except Exception as e:
                st.error(f"❌ Errore aggiornamento: {str(e)}")
                
elif page == "Foto - Aggiungi SKUs":
    new_sku = st.session_state.get("aggiunta_confermata", set())

    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(foto_sheet_id, cache_key=cache_token)
    
    # Aggiungi nuova SKU
    st.subheader("➕ Aggiungi nuova SKU")
    if st.session_state.get("aggiunta_confermata"):
        sku_added = st.session_state["aggiunta_confermata"]
        success = st.success(f"✅ SKU Aggiunta con successo: {sku_added}")
        time.sleep(2)
        success.empty();
        st.session_state["aggiunta_confermata"] = False
        st.session_state["refresh_foto_token"] = str(time.time())
        st.session_state.input_sku = ""
        st.rerun()
    else:
        add_sku_input = st.text_input("Aggiungi una nuova SKU", key="input_sku")
        new_sku = add_sku_input.upper()
        if add_sku_input:
            if new_sku not in df["SKU"].values.tolist():
                aggiungi_sku(foto_sheet_id, new_sku)
                st.session_state["aggiunta_confermata"] = add_sku_input.strip().upper()
                st.rerun()
            else:
                warning = st.warning(f"SKU {new_sku} già presente in lista")
                time.sleep(2)
                warning.empty()
                
                
elif page == "Foto - Storico":
    st.header("📚 Storico Articolo")
    st.markdown("Inserisci una SKU per visualizzare tutte le immagini storiche salvate su Dropbox per quell’articolo.")

    sku_query = st.text_input("🔎 Inserisci SKU", key="storico_sku_input")

    if sku_query:
        sku_query = sku_query.strip().upper()
        try:
            folder_path = f"/repository/{sku_query}"
            access_token = get_dropbox_access_token()
            dbx = dropbox.Dropbox(access_token)
            image_files = []
            try:
                files = dbx.files_list_folder(folder_path).entries
                image_files = [f for f in files if f.name.lower().endswith(".jpg")]
            except dropbox.exceptions.ApiError as e:
                if (isinstance(e.error, dropbox.files.ListFolderError) and
                    e.error.is_path() and
                    e.error.get_path().is_not_found()):
                    image_files = []
                else:
                    st.error(f"⚠️ Errore Dropbox: {e}")
                    image_files = []

            if not image_files:
                st.warning("❌ Nessuna immagine storica trovata su Dropbox per questa SKU.")
            
                # Mostra immagine attuale da repository.falc.biz
                image_url = f"https://repository.falc.biz/fal001{sku_query.lower()}-1.jpg"
                try:
                    resp = requests.get(image_url, timeout=5)
                    if resp.status_code == 200:
                        st.image(image_url, width=300, caption="📅 Attuale (da repository)")
                    else:
                        st.error("⚠️ Nemmeno la foto attuale è disponibile sul repository.")
                except Exception as e:
                    st.error(f"⚠️ Errore nel caricamento della foto attuale: {e}")
            else:
                def extract_date(filename):
                    base = filename.replace(".jpg", "")
                    parts = base.split("_")
                    return parts[1] if len(parts) == 2 else "Attuale"

                image_infos = [{
                    "name": f.name,
                    "path": f.path_display,
                    "date": extract_date(f.name)
                } for f in image_files]

                image_infos.sort(key=lambda x: (x["date"] != "Attuale", x["date"]))

                st.markdown(f"📸 **{len(image_infos)} immagini trovate**")

                cols = st.columns(4)
                for idx, info in enumerate(image_infos):
                    with cols[idx % 4]:
                        headers = {
                            "Authorization": f"Bearer {access_token}",
                            "Dropbox-API-Arg": json.dumps({"path": info["path"]})
                        }
                        resp = requests.post("https://content.dropboxapi.com/2/files/download", headers=headers)
                        if resp.status_code == 200:
                            st.image(resp.content, use_container_width=True)
                            st.caption(f"📅 {info['date']}")
                        else:
                            st.warning(f"⚠️ Errore immagine: {info['name']}")
        except Exception as e:
            st.error(f"Errore: {str(e)}")

elif page == "Foto - Aggiungi prelevate":
    st.header("Aggiungi prelevate")
    st.markdown("Aggiungi la lista delle paia prelevate")
    
    sheet = get_sheet(foto_sheet_id, "PRELEVATE")
    sheet_len = len(pd.DataFrame(sheet.get_all_values()))
    oggi = datetime.today().date().strftime('%d/%m/%Y')
    text_input = st.text_area("Lista paia prelevate", height=400, width=800)
    
    if text_input:
        # Regex per SKU: 7 numeri, spazio, 2 numeri, spazio, 4 caratteri alfanumerici
        #pattern = r"\b\d{7} \d{2} [A-Z0-9]{4}\b"
        pattern = r"\b[0-9a-zA-Z]{7} [0-9a-zA-Z]{2} [0-9a-zA-Z]{4}\b"
        skus_raw = re.findall(pattern, text_input)
    
        # Rimuovi spazi interni e converti in stringa (senza apostrofo per confronto)
        skus_clean = [str(sku.replace(" ", "")) for sku in skus_raw]
    
        st.subheader(f"SKU trovate: {len(skus_clean)}")
    
        if st.button("Carica su GSheet"):
            # Leggi SKU già presenti nel foglio
            existing_skus = sheet.col_values(1)
            # Rimuovi eventuali apostrofi e converti in str per confronto
            existing_skus_clean = [str(sku).lstrip("'") for sku in existing_skus]
    
            # Filtra SKU nuove
            skus_to_append_clean = [sku for sku in skus_clean if sku not in existing_skus_clean]
    
            if skus_to_append_clean:
                # Aggiungi apostrofo solo al momento dell'append per forzare formato testo
                rows_to_append = [[f"'{sku}", "=REPO(A33)", f"{oggi}", f"=SE(VAL.NON.DISP(CONFRONTA(INDIRETTO(\"LISTA!$D\"&CONFRONTA($A{sheet_len+1};LISTA!A:A;0));SPLIT(SETTINGS(\"brandMatias\");\",\");0));SE(VAL.NON.DISP(CONFRONTA(INDIRETTO(\"LISTA!$D\"&CONFRONTA($A{sheet_len+1};LISTA!A:A;0));SPLIT(SETTINGS(\"brandMatteo\");\",\");0));\"\";\"MATTEO\");\"MATIAS\")"] for sku in skus_to_append_clean]
                
                # Append a partire dall'ultima riga disponibile
                sheet.append_rows(rows_to_append, value_input_option="USER_ENTERED")
                st.success(f"✅ {len(skus_to_append_clean)} nuove SKU aggiunte al foglio PRELEVATE!")
            else:
                st.info("⚠️ Tutte le SKU inserite sono già presenti nel foglio.")
                
elif page == "Giacenze - Importa":
    st.header("Importa giacenze")

    
    options = ["Manuale", "UBIC", "PIM"]
    
    selected = option_menu(
        menu_title=None,
        options=["Manuale", "UBIC", "PIM"],
        icons=[" ", " ", " "],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "0 10px 0 0!important",
                "background-color": "#f0f0f0",
                "display": "flex",
                "justify-content": "center"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "5px",
                "padding": "0px",
                "min-height": "40px",
                "height": "40px",
                "line-height": "normal",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "box-sizing": "border-box",
                "--hover-color": "#e0e0e0",
                "before": "none"
            },
            "nav-link-selected": {
                "background-color": "#4CAF50",
                "color": "white",
                "border": "2px solid #cccccc",
                "border-radius": "10px",
                "padding": "0px",
                "min-height": "40px",
                "height": "40px",
                "line-height": "normal",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
                "box-sizing": "border-box",
                "before": "none"
            },
        }
    )

    st.session_state.selected_option = selected
    nome_file = st.session_state.selected_option

    # --- Reset se cambio file/target ---
    if "downloaded_file_name" not in st.session_state or st.session_state.downloaded_file_name != nome_file:
        st.session_state.df_input = None
        st.session_state.downloaded_file = None
        st.session_state.downloaded_file_metadata = None
        st.session_state.downloaded_file_name = nome_file

    csv_import = None
    file_bytes_for_upload = None
    last_update = None

    dbx = get_dropbox_client()
    folder_path = "/GIACENZE"

    # --- Manuale ---
    if nome_file == "Manuale":
        uploaded_file = st.file_uploader("Carica un file CSV manualmente", type="csv", key="uploader_manual")
        if uploaded_file:
            if ("uploaded_file_name" not in st.session_state) or (st.session_state.uploaded_file_name != uploaded_file.name):
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.df_input = None  # reset DataFrame se file nuovo
                st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
                uploaded_file.seek(0)
                
            csv_import = uploaded_file
            file_bytes_for_upload = st.session_state.uploaded_file_bytes
            manual_nome_file = uploaded_file.name.upper()

    # --- UBIC / PIM ---
    else:
        if st.session_state.downloaded_file is None:
            with st.spinner(f"Download {nome_file} da DropBox..."):
                st.session_state.downloaded_file, st.session_state.downloaded_file_metadata = download_csv_from_dropbox(
                    dbx, folder_path, f"{nome_file}.csv")
                st.session_state.downloaded_file_name = nome_file

        latest_file = st.session_state.downloaded_file
        metadata = st.session_state.downloaded_file_metadata
        
        if latest_file:
            csv_import = latest_file
            file_bytes_for_upload = latest_file.getvalue()
            last_update = format_dropbox_date(metadata.client_modified)
            st.info(f"{nome_file} ultimo aggiornamento: {last_update}")
        else:
            st.warning(f"Nessun file trovato su Dropbox, carica manualmente")

    # --- Carico CSV solo se df_input è None ---
    if csv_import and st.session_state.df_input is None:
        with st.spinner("Carico il CSV..."):
            st.session_state.df_input = read_csv_auto_encoding(csv_import, "\t")

    df_input = st.session_state.df_input

    default_sheet_id = foto_sheet_id
    selected_sheet_id = st.text_input("Inserisci ID del Google Sheet", value=default_sheet_id)

    col1, col2, col3, col4 = st.columns(4)
    
    if df_input is not None:
        view_df = st.checkbox("Visualizza il dataframe?", value=False)
        if view_df:
            st.write(df_input)

        # --- Colonne numeriche ---
        numeric_cols_info = { "D": "0", "L": "000", "N": "0", "O": "0" }
        for i in range(17, 32):  # Q-AE
            col_letter = gspread.utils.rowcol_to_a1(1, i)[:-1]
            numeric_cols_info[col_letter] = "0"

        def to_number_safe(x):
            try:
                if pd.isna(x) or x == "":
                    return ""
                return float(x)
            except:
                return str(x)

        for col_letter in numeric_cols_info.keys():
            col_idx = gspread.utils.a1_to_rowcol(f"{col_letter}1")[1] - 1
            if df_input.columns.size > col_idx:
                col_name = df_input.columns[col_idx]
                df_input[col_name] = df_input[col_name].apply(to_number_safe)

        target_indices = [gspread.utils.a1_to_rowcol(f"{col}1")[1] - 1 for col in numeric_cols_info.keys()]
        for idx, col_name in enumerate(df_input.columns):
            if idx not in target_indices:
                df_input[col_name] = df_input[col_name].apply(lambda x: "" if pd.isna(x) else str(x))

        data_to_write = [df_input.columns.tolist()] + df_input.values.tolist()

        # --- Destinazione GSheet ---       
        with col2:
            if st.button("Importa Giacenze"):
                sheet_upload_giacenze = get_sheet(selected_sheet_id, "GIACENZE")
                sheet_upload_pim = get_sheet(selected_sheet_id, "PIM")
                
                with st.spinner("Aggiorno giacenze su GSheet..."):
                    if nome_file == "UBIC":
                        sheet_upload_giacenze.clear()
                        sheet_upload_giacenze.update("A1", data_to_write)
                    elif nome_file == "PIM":
                        sheet_upload_pim.clear()
                        sheet_upload_pim.update("A1", data_to_write)
                    elif nome_file == "Manuale":
                        if manual_nome_file == "UBIC.CSV":
                            sheet_upload_giacenze.clear()
                            sheet_upload_giacenze.update("A1", data_to_write)
                        elif manual_nome_file == "PIM.CSV":
                            sheet_upload_pim.clear()
                            sheet_upload_pim.update("A1", data_to_write)
                            
                    last_row = len(df_input) + 1
    
                    ranges_to_format = [
                        (f"{col_letter}2:{col_letter}{last_row}",
                            CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern=pattern)))
                        for col_letter, pattern in numeric_cols_info.items()
                    ]
                    format_cell_ranges(sheet_upload_giacenze, ranges_to_format)
                    st.success("✅ Giacenze importate con successo!")

                if nome_file == "Manuale" and file_bytes_for_upload:
                    with st.spinner("Carico il file su DropBox..."):
                        upload_csv_to_dropbox(dbx, folder_path, f"{manual_nome_file}", file_bytes_for_upload)
                        
        with col3:
            if st.button("Importa Giacenze & Anagrafica"):
                sheet_upload_giacenze = get_sheet(selected_sheet_id, "GIACENZE")
                sheet_upload_pim = get_sheet(selected_sheet_id, "PIM")
                sheet_upload_anagrafica = get_sheet(selected_sheet_id, "ANAGRAFICA")
                sheet_anagrafica = get_sheet(anagrafica_sheet_id, "ANAGRAFICA")
                
                with st.spinner("Aggiorno giacenze e anagrafica su GSheet..."):
                    if nome_file == "UBIC":
                        sheet_upload_giacenze.clear()
                        sheet_upload_giacenze.update("A1", data_to_write)
                    elif nome_file == "PIM":
                        sheet_upload_pim.clear()
                        sheet_upload_pim.update("A1", data_to_write)
                    elif nome_file == "Manuale":
                        if manual_nome_file == "UBIC.CSV":
                            sheet_upload_giacenze.clear()
                            sheet_upload_giacenze.update("A1", data_to_write)
                        elif manual_nome_file == "PIM.CSV":
                            sheet_upload_pim.clear()
                            sheet_upload_pim.update("A1", data_to_write)
                            
                    last_row = len(df_input) + 1
    
                    ranges_to_format = [
                        (f"{col_letter}2:{col_letter}{last_row}",
                            CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern=pattern)))
                        for col_letter, pattern in numeric_cols_info.items()
                    ]
                    format_cell_ranges(sheet_upload_giacenze, ranges_to_format)

                    sheet_upload_anagrafica.clear()
                    sheet_upload_anagrafica.update("A1", sheet_anagrafica.get_all_values())
                    st.success("✅ Giacenze e anagrafica importate con successo!")
    
                if nome_file == "Manuale" and file_bytes_for_upload:
                    with st.spinner("Carico il file su DropBox..."):
                        upload_csv_to_dropbox(dbx, folder_path, f"{manual_nome_file}", file_bytes_for_upload)

        with col4:
            if nome_file == "Manuale" and file_bytes_for_upload:
                if st.button("Carica su DropBox"):
                    with st.spinner("Carico il file su DropBox..."):
                        upload_csv_to_dropbox(dbx, folder_path, f"{manual_nome_file}", file_bytes_for_upload)
                    
    with col1:
        if st.button("Importa Anagrafica"):
            sheet_upload_anagrafica = get_sheet(selected_sheet_id, "ANAGRAFICA")
            sheet_anagrafica = get_sheet(anagrafica_sheet_id, "ANAGRAFICA")
            
            with st.spinner("Aggiorno anagrafica su GSheet..."):
                sheet_upload_anagrafica.clear()
                sheet_upload_anagrafica.update("A1", sheet_anagrafica.get_all_values())
                st.success("✅ Anagrafica importata con successo!")
            
elif page == "Giacenze - Per corridoio":
    st.header("Riepilogo per corridoio")

    # Calcolo anno e stagione di default
    oggi = datetime.now()
    anno_default = oggi.year
    mese = oggi.month
    stagione_default = 1 if mese in [1, 2, 11, 12] else 2

    # Recupero worksheet
    sheet_id = st.secrets["APP_GSHEET_ID"]
    worksheet = get_sheet(sheet_id, "GIACENZE")

    # Leggo dati dal foglio
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Conversioni iniziali
    df = df.astype(str)
    if "GIAC.UBIC" in df.columns:
        df["GIAC.UBIC"] = pd.to_numeric(df["GIAC.UBIC"], errors="coerce").fillna(0)

    # --- Mappatura categoria adulto/bambino ---
    marchi_categoria = { 
        "NATURINO CLASSIC": "BAMBINO",
        "NATURINO WILD LIFE": "BAMBINO",
        "NATURINO ACTIVE": "BAMBINO",
        "FLOWER M.BY NATURINO": "BAMBINO",
        "FLOWER MOUNTAIN": "ADULTO",
        "VOILE BLANCHE": "ADULTO",
        "NATURINO BAREFOOT": "BAMBINO",
        "FALCOTTO ACTIVE": "BAMBINO",
        "FALCOTTO CLASSIC": "BAMBINO",
        "NATURINO EASY": "BAMBINO",
        "NATURINO COCOON": "BAMBINO",
        "FALCOTTO SNEAKERS": "BAMBINO",
        "NATURINO SNEAKERS": "BAMBINO",
        "W6YZ Adulto": "ADULTO",
        "W6YZ Bimbo": "BAMBINO",
        "NATURINO OUTDOOR": "BAMBINO",
        "Candice Cooper": "ADULTO",
        "NATURINO BABY": "BAMBINO",
        "C N R": "ADULTO" 
    }
    df["CATEGORIA"] = df["COLLEZIONE"].map(marchi_categoria)

    # Estrazione anno e stagione dalla colonna STAG
    df[["anno_stag", "stag_stag"]] = df["STAG"].str.split("/", expand=True)
    df["anno_stag"] = pd.to_numeric(df["anno_stag"], errors="coerce").fillna(0).astype(int)
    df["stag_stag"] = pd.to_numeric(df["stag_stag"], errors="coerce").fillna(0).astype(int)

    # Filtro CORR e Y ai valori consentiti
    df["CORR_NUM"] = pd.to_numeric(df["CORR"], errors="coerce")
    df = df[df["CORR_NUM"].between(1, 14)]
    df = df[df["Y"].isin(["1", "2", "3", "4"])]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Input utente
        anno = st.number_input("Anno", min_value=2000, max_value=2100, value=anno_default, step=1)
        stagione = st.selectbox("Stagione", options=[1, 2], index=[1, 2].index(stagione_default))

        # --- FILTRO CON CHECKBOX SULLA COLONNA "Y" ---
        st.subheader("Filtra valori colonna Y")
        valori_Y = sorted(df["Y"].unique())
        cols = st.columns(4)
        selezione_Y = {val: cols[i % 4].checkbox(val, value=True) for i, val in enumerate(valori_Y)}

        # --- FILTRO CON CHECKBOX SULLA COLONNA "X" ---
        st.subheader("Filtra valori colonna X")
        corr_values = sorted(df["CORR_NUM"].dropna().astype(int).unique())
        cols_corr = st.columns(4)
        selezione_corr = {c: cols_corr[i % 4].checkbox(str(c), value=True, key=f"checkbox_corr_{c}") for i, c in enumerate(corr_values)}

        # --- Checkbox filtro categoria ---
        st.subheader("Filtra per Categoria")
        categorie = ["ADULTO", "BAMBINO"]
        cols_cat = st.columns(2)
        selezione_cat = {cat: cols_cat[i % 2].checkbox(cat, value=True, key=f"checkbox_cat_{cat}") for i, cat in enumerate(categorie)}

    # --- Filtri raggruppati ---
    df_filtrato = df[
        df["Y"].isin([v for v, sel in selezione_Y.items() if sel]) &
        df["CORR_NUM"].isin([c for c, sel in selezione_corr.items() if sel]) &
        df["CATEGORIA"].isin([c for c, sel in selezione_cat.items() if sel])
    ]

    with col3:
        # Calcolo riepilogo
        results = []
        for corr_value in sorted(df_filtrato["CORR_NUM"].unique()):
            corr_df = df_filtrato[df_filtrato["CORR_NUM"] == corr_value]
            cond_vecchio = (corr_df["anno_stag"] < anno) | ((corr_df["anno_stag"] == anno) & (corr_df["stag_stag"] < stagione))
            cond_nuovo = ~cond_vecchio

            vecchio = corr_df.loc[cond_vecchio, "GIAC.UBIC"].sum()
            nuovo = corr_df.loc[cond_nuovo, "GIAC.UBIC"].sum()

            results.append({"CORR": corr_value, "VECCHIO": vecchio, "NUOVO": nuovo})

        result_df = pd.DataFrame(results)
        result_df["CORR"] = result_df["CORR"].astype(int)
        st.table(result_df.reset_index(drop=True))

    with col4:
        st.download_button(
            label="📥 Scarica PDF",
            data=genera_pdf(result_df, font_size=12, header_align="CENTER", text_align="CENTER", valign="MIDDLE"),
            file_name="giac_corridoio.pdf",
            mime="application/pdf"
        )

        # --- Bottone Scarica SKU ---
        mask_vecchio = (df_filtrato["anno_stag"] < anno) | ((df_filtrato["anno_stag"] == anno) & (df_filtrato["stag_stag"] < stagione))
        cols_export = ["CODICE", "VAR", "COLORE", "COLLEZIONE.1", "CORR", "LATO", "X", "Y", "SKU NO TGL", "GIAC.UBIC"]
        df_sku = df_filtrato.loc[mask_vecchio, cols_export].copy()
        giac_per_sku = df_sku.groupby("SKU NO TGL")["GIAC.UBIC"].sum().reset_index()
        giac_per_sku = giac_per_sku.rename(columns={"GIAC.UBIC": "GIACENZA"})
        df_sku = df_sku.merge(giac_per_sku, on="SKU NO TGL", how="left")
        df_sku = df_sku.drop_duplicates(subset=["SKU NO TGL"])
        df_sku['CORR'] = pd.to_numeric(df_sku['CORR'], errors='coerce').fillna(0).astype(int)
        df_sku['X'] = pd.to_numeric(df_sku['X'], errors='coerce').fillna(0).astype(int)
        df_sku['Y'] = pd.to_numeric(df_sku['Y'], errors='coerce').fillna(0).astype(int)
        df_sku = df_sku.sort_values(by=["CORR","X","Y","LATO","CODICE","VAR","COLORE"])
        df_sku = df_sku[["CODICE", "VAR", "COLORE", "COLLEZIONE.1", "CORR", "LATO", "X", "Y", "GIACENZA"]]
        df_sku = df_sku.rename(columns={
            "CODICE":"COD",
            "COLORE":"COL",
            "COLLEZIONE.1":"DESCRIZIONE",
            "CORR":"COR",
            "GIACENZA":"GIAC"
        })
        
        # Definizione layout PDF
        larghezza_col = {
            "COD":50,
            "VAR":35,
            "COL":40,
            "DESCRIZIONE":250,
            "COR":35,
            "LATO":35,
            "X":25,
            "Y":25,
            "GIAC":30
        }
        align_col = {"DESCRIZIONE":"LEFT"}
        limiti_chars = {"DESCRIZIONE":35}
        
        st.download_button(
            label="📥 Scarica SKUs da togliere",
            data=genera_pdf(df_sku, font_size=10, header_align="CENTER", text_align="CENTER", valign="MIDDLE",
                            col_widths=larghezza_col, align_map=align_col, truncate_map=limiti_chars),
            file_name="sku_filtrate_vecchio.pdf",
            mime="application/pdf"
        )

        
elif page == "Giacenze - Per corridoio/marchio":
    st.header("Riepilogo per corridoio e marchio")

    # --- Calcolo anno e stagione di default ---
    oggi = datetime.now()
    anno_default, mese = oggi.year, oggi.month
    stagione_default = 1 if mese in [1,2,11,12] else 2

    # --- Recupero worksheet ---
    sheet_id = st.secrets["APP_GSHEET_ID"]
    worksheet = get_sheet(sheet_id, "GIACENZE")
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.astype(str)

    # --- Conversioni numeriche e pulizie ---
    if "GIAC.UBIC" in df.columns:
        df["GIAC.UBIC"] = pd.to_numeric(df["GIAC.UBIC"], errors="coerce").fillna(0)
    df[["anno_stag","stag_stag"]] = df["STAG"].str.split("/", expand=True)
    df["anno_stag"] = pd.to_numeric(df["anno_stag"], errors="coerce").fillna(0).astype(int)
    df["stag_stag"] = pd.to_numeric(df["stag_stag"], errors="coerce").fillna(0).astype(int)
    df["CORR_NUM"] = pd.to_numeric(df["CORR"], errors="coerce")
    df = df[df["CORR_NUM"].between(1,14)]
    df = df[df["Y"].isin(["1","2","3","4"])]

    # --- Normalizzazione marchi ---
    marchi_mapping = {
        "NATURINO CLASSIC":"NATURINO","NATURINO WILD LIFE":"NATURINO","NATURINO ACTIVE":"NATURINO",
        "FLOWER M.BY NATURINO":"FM FOR NATURINO","FLOWER MOUNTAIN":"FLOWER MOUNTAIN","VOILE BLANCHE":"VOILE BLANCHE",
        "NATURINO BAREFOOT":"NATURINO","FALCOTTO ACTIVE":"FALCOTTO","FALCOTTO CLASSIC":"FALCOTTO",
        "NATURINO EASY":"NATURINO","NATURINO COCOON":"NATURINO","FALCOTTO SNEAKERS":"FALCOTTO",
        "NATURINO SNEAKERS":"NATURINO","W6YZ Adulto":"W6YZ Adulto","W6YZ Bimbo":"W6YZ Bimbo",
        "NATURINO OUTDOOR":"NATURINO","Candice Cooper":"Candice Cooper","NATURINO BABY":"NATURINO","C N R":"C N R"
    }
    df["COLLEZIONE"] = df["COLLEZIONE"].str.strip()
    df["MARCHIO_STD"] = df["COLLEZIONE"].map(marchi_mapping)

    # --- Layout filtri utente ---
    with st.container():
        st.subheader("Filtri e parametri")
        col1, col2 = st.columns([2,3])
        with col1:
            anno = st.number_input("Anno", min_value=2000, max_value=2100, value=anno_default)
            stagione = st.selectbox("Stagione", options=[1,2], index=[1,2].index(stagione_default))
        with col2:
            # Filtro Y
            st.markdown("**Filtra valori colonna Y**")
            valori_Y = sorted(df["Y"].unique())
            cols_Y = st.columns(4)
            selezione_Y = {v: cols_Y[i % 4].checkbox(v, value=True) for i,v in enumerate(valori_Y)}

            # Filtro BRAND
            st.markdown("**Filtra valori colonna BRAND**")
            brands_sorted = sorted(df["MARCHIO_STD"].dropna().unique())
            cols_brand = st.columns(4)
            selezione_brand = {b: cols_brand[i % 4].checkbox(b, value=True) for i, b in enumerate(brands_sorted)}

    # --- Applico filtri ---
    df = df[df["Y"].isin([v for v, sel in selezione_Y.items() if sel])]
    df = df[df["MARCHIO_STD"].isin([b for b, sel in selezione_brand.items() if sel])]

    # --- Pivot per tabella piatta basata sui brand filtrati ---
    df_table = df.groupby(["CORR_NUM","MARCHIO_STD"]).apply(
        lambda x: pd.Series({
            "VECCHIO": x.loc[(x["anno_stag"]<anno)|((x["anno_stag"]==anno)&(x["stag_stag"]<stagione)), "GIAC.UBIC"].sum(),
            "NUOVO": x.loc[(x["anno_stag"]>anno)|((x["anno_stag"]==anno)&(x["stag_stag"]>=stagione)), "GIAC.UBIC"].sum()
        })
    ).reset_index()

    marchi_filtrati = sorted(df_table["MARCHIO_STD"].unique())
    df_table = df_table.pivot(index="CORR_NUM", columns="MARCHIO_STD", values=["VECCHIO","NUOVO"])

    # --- Ricostruzione colonne pivot in ordine dei brand filtrati ---
    col_order = []
    for b in marchi_filtrati:
        if ("VECCHIO",b) in df_table.columns and ("NUOVO",b) in df_table.columns:
            col_order.extend([("VECCHIO",b), ("NUOVO",b)])
    df_table = df_table[col_order]
    df_table.columns = [f"{col[1]}_{col[0]}" for col in df_table.columns]
    df_table = df_table.reset_index().rename(columns={"CORR_NUM":"CORR"}).fillna(0)

    # --- Costruzione AgGrid columnDefs dinamica ---
    column_defs = [{"headerName":"CORR","headerComponentParams":{
                        "template": '<div style="text-align:center; width:100%;">CORR</div>'
                    },"field":"CORR","width":60,"pinned":"left","cellStyle":{"textAlign":"center"}}]

    for brand in marchi_filtrati:
        column_defs.append({
            "headerName": brand,
            "headerComponentParams": {
                "template": f'<div style="text-align:center; width:100%;">{brand}</div>'
            },
            "children":[
                {"headerName":"VECCHIO","headerComponentParams":{
                    "template": '<div style="text-align:center; width:100%;">VECCHIO</div>'
                },"field":f"{brand}_VECCHIO","width":70,"cellStyle":{"textAlign":"center","backgroundColor":"#FFF2CC"}},
                {"headerName":"NUOVO","headerComponentParams":{
                    "template": '<div style="text-align:center; width:100%;">NUOVO</div>'
                },"field":f"{brand}_NUOVO","width":70,"cellStyle":{"textAlign":"center","backgroundColor":"#D9E1F2"}}
            ]
        })

    gridOptions = {
        "columnDefs": column_defs,
        "defaultColDef":{
            "resizable":False,
            "sortable":False,
            "filter":False,
            "wrapText":False,
            "autoHeight":True,
            "lockPosition":True,
            "cellStyle":{"textAlign":"center"}
        },
        "domLayout":"normal","suppressHorizontalScroll":False
    }

    # --- Visualizzazione tabella ---
    st.subheader("Tabella completa per corridoio e marchio")
    AgGrid(df_table, gridOptions=gridOptions, allow_unsafe_jscode=True, height=445, fit_columns_on_grid_load=True)

    # --- Bottone PDF ---
    with col2:
        st.download_button(
            label="📥 Scarica PDF",
            data=genera_pdf_aggrid(df_table),
            file_name="giac_corridoio_marchio.pdf",
            mime="application/pdf"
        )

elif page == "Giacenze - Aggiorna anagrafica":
    st.header("Aggiorna anagrafica da CSV")

    sheet = get_sheet(anagrafica_sheet_id, "DATA")
    
    uploaded_file = st.file_uploader("Carica CSV", type=["csv"])

    if uploaded_file:
        if st.button("Carica su GSheet"):
            added, updated = process_csv_and_update(sheet, uploaded_file)
            st.success(f"✅ Aggiunte {added} nuove SKU, aggiornate {updated} SKU già presenti.")

    
elif page == "Logout":
    logout()

elif page == "Giacenze - Old import":
    st.header("Importa giacenze")
    st.markdown("Importa le giacenze da file CSV.")
    
    sheet_id = st.secrets["APP_GSHEET_ID"]
    sheet = get_sheet(sheet_id, "GIACENZE")
    csv_import = st.file_uploader("Carica un file CSV", type="csv")
    
    if csv_import:
        df_input = read_csv_auto_encoding(csv_import, "\t")
    
        # Lista delle colonne da formattare come numeriche con pattern
        numeric_cols_info = {
            "D": "0",
            "L": "000",
            "N": "0",
            "O": "0",
        }

        # Colonne Q-AE
        for i in range(17, 32):  # Q=17, AE=31
            col_letter = gspread.utils.rowcol_to_a1(1, i)[:-1]  # togli solo il numero finale
            numeric_cols_info[col_letter] = "0"

        # Funzione bulletproof: converte solo valori numerici, testo rimane testo
        def to_number_safe(x):
            try:
                if pd.isna(x) or x == "":
                    return ""
                return float(x)
            except:
                return str(x)
    
        # Applica conversione solo alle colonne target
        for col_letter in numeric_cols_info.keys():
            col_idx = gspread.utils.a1_to_rowcol(f"{col_letter}1")[1] - 1  # indice zero-based
            if df_input.columns.size > col_idx:
                col_name = df_input.columns[col_idx]
                df_input[col_name] = df_input[col_name].apply(to_number_safe)

        # Tutte le altre colonne → forzale a stringa per evitare conversioni indesiderate
        target_indices = [gspread.utils.a1_to_rowcol(f"{col}1")[1] - 1 for col in numeric_cols_info.keys()]
        test = []
        for idx, col_name in enumerate(df_input.columns):
            if idx not in target_indices:
                test.append(idx)
                df_input[col_name] = df_input[col_name].apply(lambda x: "" if pd.isna(x) else str(x))

        # Trasforma tutto in lista per Google Sheet
        data_to_write = [df_input.columns.tolist()] + df_input.values.tolist()
    
        st.write(df_input)
    
        if st.button("Importa"):
            sheet.clear()
            sheet.update("A1", data_to_write)
            last_row = len(df_input) + 1  # +1 per intestazione
    
            # Prepara la lista di range da formattare
            ranges_to_format = []
            for col_letter, pattern in numeric_cols_info.items():
                ranges_to_format.append(
                    (f"{col_letter}2:{col_letter}{last_row}",
                     CellFormat(numberFormat=NumberFormat(type="NUMBER", pattern=pattern)))
                )
    
            # Applica il formato in un colpo solo
            format_cell_ranges(sheet, ranges_to_format)
    
            st.success("✅ Giacenze importate con successo!")

elif page == "Admin":
    st.write("Sezione admin")

elif page == "Admin - Aggiungi utente":
    st.header("Aggiungi nuovo utente")
    st.markdown("Crea un nuovo utente nella tabella Supabase.")

    with st.form("register_form", clear_on_submit=True):
        nome = st.text_input("Nome", key="nome")
        cognome = st.text_input("Cognome", key="cognome")
        username = st.text_input("Username", key="username")
        email = st.text_input("Email", key="email")
        password = st.text_input("Password", type="password", key="password")
        role = st.selectbox("Ruolo",["Guest","Logistica","Customer Care","Admin"], key="role")
        #username = nome + " " + cognome
        
        submit = st.form_submit_button("Crea utente")
        if submit:
            register_user(email, password, nome=nome, cognome=cognome, username=username, role=role.lower())

elif page == "Ordini - Dashboard":
    st.title("Dashboard")
    sheet = get_sheet(ordini_sheet_id, "ORDINI")

    data = sheet.get_all_values()
    headers = data[0]

    df = pd.DataFrame(data[1:], columns=headers)

    df['Quantita'] = pd.to_numeric(df['Quantita'], errors='coerce')
    df['Data'] = pd.to_datetime(df['Data'], format="%d/%m/%Y", errors='coerce')

    all_marketplaces = ['Tutti i marketplace'] + list(df['Marketplace'].unique())
    selected_marketplace = st.selectbox('Seleziona Marketplace', all_marketplaces)

    all_countries = ['Tutte le nazioni'] + list(df['Nazione'].unique())
    selected_country = st.selectbox('Seleziona Nazione', all_countries)

    filtered_df = df.copy()
    if selected_marketplace != 'Tutti i marketplace':
        filtered_df = filtered_df[filtered_df['Marketplace'] == selected_marketplace]
    if selected_country != 'Tutte le nazioni':
        filtered_df = filtered_df[filtered_df['Nazione'] == selected_country]

    view_df = st.checkbox("Visualizza dataframe?", value=False)
    if view_df:
        st.dataframe(filtered_df)

    # ---
    st.subheader("Riepilogo Dati")
    
    col1, col2, col3 = st.columns(3)
    
    total_orders = filtered_df['Numero Ordine'].nunique()
    col1.metric("Ordini Analizzati", total_orders)

    total_items = filtered_df['Quantita'].sum()
    col2.metric("Articoli Totali Venduti", total_items)
    
    unique_marketplaces = filtered_df['Marketplace'].nunique()
    col3.metric("Marketplace Unici", unique_marketplaces)

    # ---
    st.subheader("Analisi Visuale")
    
    st.markdown("Quantità venduta per Marketplace")
    market_sales = filtered_df.groupby('Marketplace')['Quantita'].sum().reset_index()
    st.bar_chart(market_sales, x='Marketplace', y='Quantita')

    st.markdown("Quantità venduta per Nazione")
    country_sales = filtered_df.groupby('Nazione')['Quantita'].sum().reset_index()
    st.bar_chart(country_sales, x='Nazione', y='Quantita')
    

elif page == "Ordini - Importa":
    st.title("Dashboard - Analizza PDF")
    st.write("Carica un PDF con gli ordini (1 ordine per pagina) per estrarre le informazioni.")

    sheet = get_sheet(ordini_sheet_id, "ORDINI")

    uploaded_files = st.file_uploader("Scegli un file PDF", type="pdf", accept_multiple_files=True)

    all_orders_data = []
    if uploaded_files:
        with st.spinner("Analizzando i PDF..."):
            all_orders_data = []

            for uploaded_file in uploaded_files:
                reader = PyPDF2.PdfReader(uploaded_file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        order_data = extract_data_from_page(page_text)
                        if order_data['Articoli']:
                            for item in order_data['Articoli']:
                                all_orders_data.append({
                                    'Numero Ordine': order_data['Numero Ordine'],
                                    'Marketplace': order_data['Marketplace'],
                                    'Data': order_data['Data'],
                                    'Nazione': order_data['Nazione'],
                                    'Codice': item['codice'],
                                    'Taglia': item['taglia'],
                                    'Quantita': item['quantita'],
                                    'Descrizione': item['descrizione']
                                })
    
        df = pd.DataFrame(all_orders_data)
        st.write(len(all_orders_data))
        ordine_colonne = ["Data", "Marketplace", "Nazione", "Numero Ordine", "Codice", "Taglia", "Quantita"]
        
        
        col1, col2, col3 = st.columns(3)
        
        total_orders = df['Numero Ordine'].nunique()
        col1.metric("Ordini Analizzati", total_orders)
    
        df['Quantita'] = pd.to_numeric(df['Quantita'], errors="coerce")
        total_items = df['Quantita'].sum()
        col2.metric("Articoli Totali Venduti", total_items)
        
        unique_marketplaces = df['Marketplace'].nunique()
        col3.metric("Marketplace Unici", unique_marketplaces)

        data = df[ordine_colonne].values.tolist()
        if st.button("Carica su gsheet"):
            sheet.append_rows(data, value_input_option="RAW")

elif page == "Catalogo - Trova articolo":
    st.write("trova articolo")

elif page == "Catalogo - Aggiungi ordini stagione":
    st.title("Aggiungi ordini di stagione")
    st.write("Carica un CSV con gli ordini")

    uploaded_files = st.file_uploader("Aggiungi i files CSV", type=["csv"], accept_multiple_files=True)

    ecom = []
    zfs = []
    amazon = []
    if uploaded_files:
        for file in uploaded_files:
            csv = read_csv_auto_encoding(file, separatore=",")
            data = pd.DataFrame(csv)
            data = data[1:]

            if data["COD.CLIENTI"] == "0019243.016":
                ecom.append(data)
            elif  data["COD.CLIENTI"] == "0039632":
                zfs.append(data)
            elif  data["COD.CLIENTI"] == "0034630":
                amazon.append(data)

    st.write(ecom)
    st.write(zfs)
    st.write(amazon)

elif page == "Ferie - Aggiungi ferie":
    st.header("Aggiungi ferie")
    
    sheet = get_sheet(ferie_sheet_id, "FERIE")

    ferie_esistenti = sheet.get_all_values()
    users_list = supabase.table("profiles").select("*").execute().data
    utenti = [f"{u['nome']} {u['cognome']}" for u in users_list]
    id_mappa = {f"{u['nome']} {u['cognome']}": u["user_id"] for u in users_list}

    
    oggi = datetime.now().date()
    utente_selezionato = st.selectbox("Seleziona utente", utenti)
    data_inizio = st.date_input("Data inizio", oggi)
    data_fine = st.date_input(
        "Data fine", 
        value=max(data_inizio, oggi), 
        min_value=data_inizio
    )
    motivazione = st.text_input("Motivazione")

    if st.button("Aggiungi ferie"):
        data = [utente_selezionato, data_inizio.strftime('%d/%m/%Y'), data_fine.strftime('%d/%m/%Y'), motivazione]
        sheet.append_row(data, value_input_option="USER_ENTERED")

elif page == "Ferie - Report":
    st.header("📅 Report ferie settimanale")
    
    # 1. Leggi dati ferie da GSheet
    sheet = get_sheet(ferie_sheet_id, "FERIE")
    ferie_data = sheet.get_all_values()
    ferie_df = pd.DataFrame(
        ferie_data[1:], columns=ferie_data[0]
    ) if len(ferie_data) > 1 else pd.DataFrame(columns=["NOME", "DATA INIZIO", "DATA FINE", "MOTIVO"])
    
    # 2. Selettore giorno iniziale
    today = datetime.now().date()
    start_date = st.date_input("Seleziona il giorno di inizio settimana", value=today)
    days_of_week = [start_date + timedelta(days=i) for i in range(7)]
    
    # Mappa abbreviata giorni in italiano
    giorni_settimana_it = {
        "Mon": "Lun",
        "Tue": "Mar",
        "Wed": "Mer",
        "Thu": "Gio",
        "Fri": "Ven",
        "Sat": "Sab",
        "Sun": "Dom"
    }
    days_labels = [giorni_settimana_it[day.strftime("%a")] + day.strftime(" %d/%m") for day in days_of_week]
    
    # 3. Prepara lista utenti
    users_list = supabase.table("profiles").select("*").execute().data
    utenti = sorted([f"{u['nome']} {u['cognome']}" for u in users_list])
    
    # 4. Costruisci matrice ferie (utente x giorno)
    ferie_matrix = []
    for utente in utenti:
        row = []
        ferie_utente = ferie_df[ferie_df["NOME"] == utente]
        ferie_utente = ferie_utente.copy()
        for idx, r in ferie_utente.iterrows():
            try:
                ferie_utente.at[idx, "DATA INIZIO"] = datetime.strptime(r["DATA INIZIO"], "%d/%m/%Y").date()
                ferie_utente.at[idx, "DATA FINE"] = datetime.strptime(r["DATA FINE"], "%d/%m/%Y").date()
            except Exception:
                ferie_utente.at[idx, "DATA INIZIO"] = None
                ferie_utente.at[idx, "DATA FINE"] = None
    
        for giorno in days_of_week:
            in_ferie = False
            motivo = ""
            for _, r in ferie_utente.iterrows():
                if r["DATA INIZIO"] and r["DATA FINE"] and r["DATA INIZIO"] <= giorno <= r["DATA FINE"]:
                    in_ferie = True
                    motivo = r.get("MOTIVO", "")
            if in_ferie:
                if motivo == "Malattia":
                    row.append("🇨🇭" + (f" {motivo}" if motivo else ""))
                else:
                    row.append("🌴" + (f" {motivo}" if motivo else ""))
            else:
                row.append("")
        ferie_matrix.append(row)
    
    # 5. Visualizza tabella con celle evidenziate e centrata
    ferie_report_df = pd.DataFrame(ferie_matrix, columns=days_labels, index=utenti)
    
    def evidenzia_ferie(val):
        if isinstance(val, str) and val.startswith("🌴"):
            return 'background-color: #E6F7DD; text-align: center;'
        elif isinstance(val, str) and val.startswith("🇨🇭"):
            return 'background-color: #FFA1A1; text-align: center;'
        return 'text-align: center;'
    
    ferie_report_df_styled = (
        ferie_report_df
        .style
        .applymap(evidenzia_ferie)
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "normal"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
        .set_properties(**{"text-align": "center"})
    )
    
    st.markdown("""
        <style>
            .streamlit-expander, .block-container, table {
                margin-left: auto !important;
                margin-right: auto !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(ferie_report_df_styled.to_html(escape=False), unsafe_allow_html=True)
