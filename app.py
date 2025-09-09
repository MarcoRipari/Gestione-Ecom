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
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
import gspread.utils
import re
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
from reportlab.lib.pagesizes import landscape
import html
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from googleapiclient.discovery import build
import io
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.http import MediaInMemoryUpload
from dateutil import parser
from dateutil.tz import tzlocal
import locale
from zoneinfo import ZoneInfo
from supabase import create_client, Client
from gspread.utils import rowcol_to_a1
import pdfplumber
import PyPDF2

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

# ---------------------------
# Auth system
# ---------------------------
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
service_role_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)
supabase_admin = create_client(supabase_url, service_role_key)

def login(username: str, password: str) -> bool:
    try:
        # 1. Recupera il profilo dallo username
        res_profile = supabase.table("profiles").select("*").eq("username", username).single().execute()
        if not res_profile.data:
            st.error("❌ Username non trovato")
            return False

        user_id = res_profile.data["user_id"]

        # 2. Recupera l'utente auth per ottenere l'email (richiede service_role_key)
        res_user = supabase_admin.auth.admin.get_user_by_id(user_id)
        email = res_user.user.email

        if not email:
            st.error("❌ Nessuna email trovata per questo utente")
            return False

        # 3. Login usando email + password
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if res.user:
            # 4. Salva in session_state
            st.session_state.user = {
                "email": email,
                "username": res_profile.data.get("username", ""),
                "nome": res_profile.data.get("nome", ""),
                "cognome": res_profile.data.get("cognome", ""),
                "role": res_profile.data.get("role", "guest"),
            }
            return True
        else:
            st.error("❌ Credenziali errate")
            return False

    except Exception as e:
        st.error(f"Errore login: {e}")
        return False


      
def login_password(email: str, password: str) -> bool:
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if res.user is not None:
           
            # Recupera il profilo dell'utente usando user_id
            profile = supabase.table("profiles").select("*").eq("user_id", res.user.id).single().execute()
            
            if profile.data is None:
                st.error("❌ Profilo utente non trovato")
                return False
            
            # Salva tutto in session_state
            st.session_state.user = {
                "data": res.user,
                "email": res.user.email,
                "nome": profile.data["nome"],
                "cognome": profile.data["cognome"],
                "username": profile.data["username"],
                "role": profile.data["role"]
            }
            #st.session_state.user = res.user
            #st.session_state.username = profile.data.get("username", res.user.email)
            return True
        else:
            st.error("❌ Email o password errati")
            return False
    except Exception as e:
        st.error(f"Errore login: {e}")
        return False


def logout():
    if "user" in st.session_state:
        supabase.auth.sign_out()
        st.session_state.user = None
        #st.session_state.username = None
        st.rerun()

def register_user(email: str, password: str, **param) -> bool:
    try:
        # 1. Crea l'utente in Supabase Auth
        res = supabase_admin.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })

        if not res.user:
            st.error("❌ Errore nella creazione utente in Auth")
            return False

        user_id = res.user.id

        # 2. Inserisci il profilo nella tabella profiles
        profile = {
            "user_id": user_id,
            "nome": param.get("nome", None),
            "cognome": param.get("cognome", None),
            "username": param.get("username", None),
            "role": param.get("role", None)
        }

        supabase_admin.table("profiles").insert(profile).execute()

        st.success(f"✅ Utente {username} creato correttamente")
        return True

    except Exception as e:
        st.error(f"Errore registrazione: {e}")
        return False

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
    prompt = f"""Scrivi due descrizioni per una calzatura da vendere online (e-commerce) in ciascuna delle seguenti lingue: {lang_list}.

>>> FORMATO OUTPUT
{{"it":{{"desc_lunga":"...","desc_breve":"..."}}, "en":{{...}}, "fr":{{...}}, "de":{{...}}}}

>>> GUIDA STILE
- Tono: {", ".join(selected_tones)}
- Ometti sempre: Codice, Nome, Marca, Colore (nemmeno in forma implicita)
- Lingua: adatta al paese target
- Non usare il genere
- Usa sempre la parola strappo, niente sinonimi ne velcro
- Evita le percentuali materiali

>>> REGOLE
- desc_lunga: {desc_lunga_length} parole → enfasi su comfort, materiali, utilizzo
- desc_breve: {desc_breve_length} parole → adatta a social media o schede prodotto rapide

>>> INFO ARTICOLO
{product_info}
{image_line}

{sim_text}
"""
    return prompt
    
# ---------------------------
# 📊 Audit Trail & Google Sheets
# ---------------------------
def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    sheet.append_rows(values, value_input_option="RAW")  # ✅ chiamata unica

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
    
def calcola_tokens(df_input, col_display_names, selected_langs, selected_tones, desc_lunga_length, desc_breve_length, k_simili, use_image, faiss_index, DEBUG=False):
    if df_input.empty:
        return None, None, "❌ Il CSV è vuoto"

    row = df_input.iloc[0]

    simili = pd.DataFrame([])
    if k_simili > 0 and faiss_index:
        index, index_df = faiss_index
        simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)

    caption = get_blip_caption(row.get("Image 1", "")) if use_image and row.get("Image 1", "") else None

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
# Async
# ---------------------------
client = AsyncOpenAI(api_key=openai.api_key)

async def async_generate_description(prompt: str, idx: int):
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        content = response.choices[0].message.content
        usage = response.usage  # <-- aggiunto
        data = json.loads(content)
        return idx, {"result": data, "usage": usage.model_dump()}
    except Exception as e:
        return idx, {"error": str(e)}

async def generate_all_prompts(prompts: list[str]) -> dict:
    tasks = [async_generate_description(prompt, idx) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)

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
    sheet_lista.append_row([sku, st.session_state.user["username"]])

@st.cache_data(ttl=300)
def carica_lista_foto(sheet_id: str, cache_key: str = "") -> pd.DataFrame:
    try:
        sheet = get_sheet(sheet_id, "LISTA")
        values = sheet.get("A3:V5000")
        if not values:
            return pd.DataFrame()
        
        # ✅ Definizione corretta: 22 intestazioni per colonne A–V
        headers = ["SKU", "CANALE", "STAGIONE", "COLLEZIONE", "DESCRIZIONE", "COD", "VAR", "COL", "TG CAMP", "TG PIC", "SCATTARE", "CONTROLLO", "DISP", "RISCATTARE", "CONSEGNATA", "FOTOGRAFO", "COR", "LAT", "X", "Y", "REPO", "END"]
        df = pd.DataFrame(values, columns=headers)
        df = df[df["SKU"].notna() & (df["SKU"].str.strip() != "")]

        # 🧹 Normalizza booleani
        def normalize_bool(col):
            return col.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
        
        df["SCATTARE"] = normalize_bool(df["SCATTARE"])
        df["CONSEGNATA"] = normalize_bool(df["CONSEGNATA"])
        df["RISCATTARE"] = normalize_bool(df["RISCATTARE"])
        df["DISP"] = normalize_bool(df["DISP"])

        return df[["SKU", "STAGIONE", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE", "RISCATTARE", "CONSEGNATA", "DISP", "COD", "VAR", "COL", "TG PIC", "FOTOGRAFO", "COR", "LAT", "X", "Y"]]
    except Exception as e:
        st.error(f"Errore durante il caricamento: {str(e)}")
        return pd.DataFrame()

# ---------------------------
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Gestione ECOM", layout="wide")

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
                          {"name":"Dashboard", "icon":"bar-chart", "role":["admin"]},
                          {"name":"Descrizioni", "icon":"list", "role":["customer care","admin"]},
                          {"name":"Foto", "icon":"camera", "role":["logistica","customer care","admin"]},
                          {"name":"Giacenze", "icon":"box", "role":["logistica","customer care","admin"]},
                          {"name":"Admin", "icon":"gear", "role":["admin"]},
                          {"name":"Logout", "icon":"key", "role":["guest","logistica","customer care","admin"]}
                         ]
        
        submenu_item_list = [{"main":"Dashboard", "name":"Analizzatore PDF", "icon":"file-earmark-pdf", "role":["admin"]},
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
    st.subheader("📌 HomePage")
    st.markdown("""
    
    """)

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
    sheet_id = st.secrets["DESC_GSHEET_ID"]
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
    
            def_column = ["skuarticolo",
                          "Classification",
                          "Matiere", "Sexe",
                          "Saison", "Silouhette",
                          "shoe_toecap_zalando",
                          "shoe_detail_zalando",
                          "heel_height_zalando",
                          "heel_form_zalando",
                          "sole_material_zalando",
                          "shoe_fastener_zalando",
                          "pattern_zalando",
                          "upper_material_zalando",
                          "futter_zalando",
                          "Subtile2",
                          "Concept",
                          "Sp.feature"
                         ]
    
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
                            f"Etichetta: {col}", value=st.session_state.col_display_names[col], key=f"label_{col}"
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
    
            with settings_col2:
                selected_labels = st.multiselect(
                    "Lingue di output",
                    options=list(LANG_LABELS.keys()),
                    default=["Italiano", "Inglese", "Francese", "Tedesco"]
                )
                selected_langs = [LANG_LABELS[label] for label in selected_labels]
                
                selected_tones = st.multiselect(
                    "Tono desiderato",
                    ["professionale", "amichevole", "accattivante", "descrittivo", "tecnico", "ironico", "minimal", "user friendly", "SEO-friendly"],
                    default=["professionale", "user friendly", "SEO-friendly"]
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
    
                    
                try:
                    with st.spinner("📚 Carico storico e indice FAISS..."):
                        tab_storico = f"STORICO_{marchio}"
                        data_sheet = get_sheet(sheet_id, tab_storico)
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
                            tab_df = pd.DataFrame(get_sheet(sheet_id, lang).get_all_records())
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
                            caption = get_blip_caption(row.get("Image 1", "")) if use_image and row.get("Image 1", "") else None
                            prompt = build_unified_prompt(row, st.session_state.col_display_names, selected_langs, image_caption=caption, simili=simili)
                            all_prompts.append(prompt)
            
                    with st.spinner("🚀 Generazione asincrona in corso..."):
                        results = asyncio.run(generate_all_prompts(all_prompts))
            
                    # Parsing risultati
                    all_outputs = already_generated.copy()
                    logs = []
            
                    for i, (_, row) in enumerate(df_input_to_generate.iterrows()):
                        result = results.get(i, {})
                        if "error" in result:
                            logs.append({
                                "sku": row.get("SKU", ""),
                                "status": f"Errore: {result['error']}",
                                "prompt": all_prompts[i],
                                "output": "",
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            continue
            
                        for lang in selected_langs:
                            lang_data = result.get("result", {}).get(lang.lower(), {})
                            descr_lunga = lang_data.get("desc_lunga", "").strip()
                            descr_breve = lang_data.get("desc_breve", "").strip()
            
                            output_row = row.to_dict()
                            output_row["Description"] = descr_lunga
                            output_row["Description2"] = descr_breve
                            all_outputs[lang].append(output_row)
            
                        log_entry = {
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
                        for lang in selected_langs:
                            df_out = pd.DataFrame(all_outputs[lang])
                            df_new = df_out[df_out["SKU"].isin(df_input_to_generate["SKU"].astype(str))]
                            if not df_new.empty:
                                append_to_sheet(sheet_id, lang, df_new)
                        for log in logs:
                            append_log(sheet_id, log)
            
                    # 📦 ZIP finale
                    with st.spinner("📦 Generazione ZIP..."):
                        mem_zip = BytesIO()
                        with zipfile.ZipFile(mem_zip, "w") as zf:
                            for lang in selected_langs:
                                df_out = pd.DataFrame(all_outputs[lang])
                                df_export = pd.DataFrame({
                                    "SKU": df_out.get("SKU", ""),
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
                        if sheet_id:
                            tab_storico = f"STORICO_{marchio}"
                            data_sheet = get_sheet(sheet_id, tab_storico)
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
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
    selected_ristampe = st.session_state.get("ristampe_selezionate", set())
    
    # 🔽 Caricamento dati con chiave cache dinamica
    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(sheet_id, cache_key=cache_token)
    st.session_state["df_lista_foto"] = df

    # 1️⃣ Genero le liste per i fotografi
    df_disp = df[df["DISP"] == True]
    #df_disp = df_disp[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y","FOTOGRAFO"]]
    df_disp = df_disp.sort_values(by=["COR", "X", "Y", "LAT"])

    df_matias = df_disp[df_disp["FOTOGRAFO"] == "MATIAS"]
    df_matteo = df_disp[df_disp["FOTOGRAFO"] == "MATTEO"]

    df_matias = df_matias[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y"]]
    df_matteo = df_matteo[["COD","VAR","COL","TG PIC","DESCRIZIONE","COR","LAT","X","Y"]]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("📦 Genera lista SKU"):
            try:
                genera_lista_sku(sheet_id, tab_names)
                st.toast("✅ Lista SKU aggiornata!")
            except Exception as e:
                st.error(f"Errore: {str(e)}")
    with col3:
        if df_disp.empty:
            st.download_button(
                label="📥 Lista Matias",
                data=genera_pdf(df_matias),
                file_name="lista_disp_matias.pdf",
                mime="application/pdf",
                disabled=True
            )
        else:
            st.download_button(
                label="📥 Lista Matias",
                data=genera_pdf(df_matias),
                file_name="lista_disp_matias.pdf",
                mime="application/pdf"
            )
    with col4:
        if df_disp.empty:
            st.download_button(
                label="📥 Lista Matteo",
                data=genera_pdf(df_matteo),
                file_name="lista_disp_matteo.pdf",
                mime="application/pdf",
                disabled=True
            )
        else:
            st.download_button(
                label="📥 Lista Matteo",
                data=genera_pdf(df_matteo),
                file_name="lista_disp_matteo.pdf",
                mime="application/pdf"
            )
    with col6:
        if st.button("🔄 Refresh"):
            st.session_state["refresh_foto_token"] = str(time.time())
    
    # 📊 Riepilogo
    total = len(df)
    consegnate = df["CONSEGNATA"].sum()
    da_scattare = df["SCATTARE"].sum()
    scattate = total - da_scattare
    matias = df[df["FOTOGRAFO"] == "MATIAS"].shape[0]
    matteo = df[df["FOTOGRAFO"] == "MATTEO"].shape[0]
        
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📝 Totale SKU", total)
    c2.metric("✅ Già scattate", scattate)
    c3.metric("🚚 Dal fotografo", consegnate)
    c4.metric("📸 Da scattare", da_scattare)
    c5.metric("Disponibili Matias", matias)
    c6.metric("Disponibili Matteo", matteo)
    
   
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
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
    selected_ristampe = st.session_state.get("ristampe_selezionate", set())

    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(sheet_id, cache_key=cache_token)

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
                sheet = get_sheet(sheet_id, "LISTA")
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
        
                range_update = f"N3:N{len(nuovi_valori) + 2}"
                sheet.update(values=nuovi_valori, range_name=range_update)
        
                # 🔄 Ricarico il DataFrame dal Google Sheet (stesso metodo usato sopra)
                df = carica_lista_foto(sheet_id, cache_key=str(time.time()))
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
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
    new_sku = st.session_state.get("aggiunta_confermata", set())

    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(sheet_id, cache_key=cache_token)
    
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
                aggiungi_sku(sheet_id, new_sku)
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
    
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
    sheet = get_sheet(sheet_id, "PRELEVATE")
    
    text_input = st.text_area("Lista paia prelevate", height=400, width=800)
    
    if text_input:
        # Regex per SKU: 7 numeri, spazio, 2 numeri, spazio, 4 caratteri alfanumerici
        pattern = r"\b\d{7} \d{2} [A-Z0-9]{4}\b"
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
                rows_to_append = [[f"'{sku}"] for sku in skus_to_append_clean]
    
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
            manual_nome_file = uploaded_file.name

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

    default_sheet_id = st.secrets["FOTO_GSHEET_ID"]
    sheet_id = st.text_input("Inserisci ID del Google Sheet", value=default_sheet_id)
    sheet_upload_giacenze = get_sheet(sheet_id, "GIACENZE")
    sheet_upload_anagrafica = get_sheet(sheet_id, "ANAGRAFICA")
    sheet_anagrafica = get_sheet(st.secrets["ANAGRAFICA_GSHEET_ID"], "ANAGRAFICA")
    
    col1, col2, col3 = st.columns(3)
    
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
                with st.spinner("Aggiorno giacenze su GSheet..."):
                    sheet_upload_giacenze.clear()
                    sheet_upload_giacenze.update("A1", data_to_write)
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
                with st.spinner("Aggiorno giacenze e anagrafica su GSheet..."):
                    sheet_upload_giacenze.clear()
                    sheet_upload_giacenze.update("A1", data_to_write)
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
                    
    with col1:
        if st.button("Importa Anagrafica"):
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
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
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
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
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

    sheet_id = st.secrets["ANAGRAFICA_GSHEET_ID"]
    sheet = get_sheet(sheet_id, "DATA")
    
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
    
    sheet_id = st.secrets["FOTO_GSHEET_ID"]
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

    with st.form("register_form"):
        nome = st.text_input("Nome")
        cognome = st.text_input("Cognome")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Ruolo",["Guest","Logistica","Customer Care","Admin"])
        username = nome + " " + cognome
        
        submit = st.form_submit_button("Crea utente")
        if submit:
            register_user(email, password, nome=nome, cognome=cognome, username=username, role=role.lower())

elif page == "Dashboard - Analizzatore PDF":
    # Funzione per estrarre i dati da una pagina PDF
    def extract_data_from_page(page_text):
        data = {}
        
        # Marketplace
        #marketplace_match = re.search(r"Marketplace:\s*([a-zA-Z0-9\s.-]+)", page_text, re.IGNORECASE)
        marketplace_match = re.search(r"Marketplace:(.*)", page_text, re.IGNORECASE)
        if marketplace_match:
            data['Marketplace'] = marketplace_match.group(1).strip()
        else:
            data['Marketplace'] = "N/A"
    
        # Numero Ordine
        order_match = re.search(r"Marketplace order\s*([a-zA-Z0-9-]+)", page_text)
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
        country_match = re.search(r"\n([A-Z]{2})\s*$", page_text.strip(), re.MULTILINE)
        if country_match:
            data['Nazione'] = country_match.group(1).strip()
        else:
            data['Nazione'] = "N/A"
        
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
                    item_match = re.search(r"^\s*(\d+)\s+([^\s]+)\s+(\d+)\s+(.*)", line)
                    if item_match:
                        item_data = {
                            'quantita': item_match.group(1).strip(),
                            'codice': item_match.group(2).strip(),
                            'taglia': item_match.group(3).strip(),
                            'descrizione': item_match.group(4).strip()
                        }
                        items.append(item_data)
                    else:
                        # Case 2: Code Size Quantity Description
                        item_match = re.search(r"^\s*([^\s]+)\s+(\d+)\s+(\d+)\s+(.*)", line)
                        if item_match:
                            item_data = {
                                'codice': item_match.group(1).strip(),
                                'taglia': item_match.group(2).strip(),
                                'quantita': item_match.group(3).strip(),
                                'descrizione': item_match.group(4).strip()
                            }
                            items.append(item_data)
                            
        data['Articoli'] = items
        
        return data
    
    # Funzione principale per la sezione 'Analizza PDF'
    def pdf_analyzer_section():
        st.title("Dashboard - Analizza PDF")
        st.write("Carica un PDF con gli ordini (1 ordine per pagina) per estrarre le informazioni.")
    
        uploaded_file = st.file_uploader("Scegli un file PDF", type="pdf")
        
        if uploaded_file is not None:
            st.success("File caricato con successo!")
            
            # Inizializza il lettore PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)
            
            st.write(f"Numero di pagine trovate: {num_pages}")
            
            extracted_data = []
            for page_num in range(num_pages):
                page_obj = pdf_reader.pages[page_num]
                page_text = page_obj.extract_text()
                
                # Controlla se la pagina è un ordine
                if "ORDINE ECOMMERCE" in page_text:
                    #st.write(f"Analizzo la pagina {page_num + 1}...")
                    data = extract_data_from_page(page_text)
                    extracted_data.append(data)
                    
            if extracted_data:
                st.subheader("Dati Estratti:")
                
                # Creazione del DataFrame
                data_for_df = []
                for order in extracted_data:
                    for item in order['Articoli']:
                        row = {
                            'Numero Ordine': order['Numero Ordine'],
                            'Marketplace': order['Marketplace'],
                            'Data': order['Data'],
                            'Nazione': order['Nazione'],
                            'Quantita': int(item.get('quantita', 0)),
                            'Codice': item.get('codice', 'N/A'),
                            'Descrizione': item.get('descrizione', 'N/A'),
                            'Taglia': item.get('taglia', 'N/A')
                        }
                        data_for_df.append(row)
                
                df = pd.DataFrame(data_for_df)
                
                st.write("Ecco i dati estratti nel DataFrame:")
                st.dataframe(df)
    
                # ---
                st.subheader("Riepilogo Dati")
                
                col1, col2, col3 = st.columns(3)
                
                total_orders = df['Numero Ordine'].nunique()
                col1.metric("Ordini Analizzati", total_orders)
    
                total_items = df['Quantita'].sum()
                col2.metric("Articoli Totali Venduti", total_items)
                
                unique_marketplaces = df['Marketplace'].nunique()
                col3.metric("Marketplace Unici", unique_marketplaces)
    
                # ---
                st.subheader("Analisi Visuale")
                
                st.markdown("Quantità venduta per Marketplace")
                market_sales = df.groupby('Marketplace')['Quantita'].sum().reset_index()
                st.bar_chart(market_sales, x='Marketplace', y='Quantita')
    
                st.markdown("Quantità venduta per Nazione")
                country_sales = df.groupby('Nazione')['Quantita'].sum().reset_index()
                st.bar_chart(country_sales, x='Nazione', y='Quantita')
                
            else:
                st.warning("Nessun ordine trovato nel PDF caricato.")

    pdf_analyzer_section()
