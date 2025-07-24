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
import aiohttp
import json
from openai import AsyncOpenAI
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_fixed

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

def generate_descriptions(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=3000
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
    sheet.append_rows(values, value_input_option="RAW")  # ✅ chiamata unica
    
# ---------------------------
# Funzioni varie
# ---------------------------
def read_csv_auto_encoding(uploaded_file):
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] or 'utf-8'
    uploaded_file.seek(0)  # Rewind after read
    return pd.read_csv(uploaded_file, encoding=encoding)

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

async def check_single_photo(session, sku: str) -> tuple[str, bool]:
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    try:
        async with session.head(url, timeout=10) as response:
            return sku, response.status != 200  # True = manca
    except:
        return sku, True  # Considera mancante in caso di errore

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))  # Retry automatico per errori temporanei
async def check_single_photo(session, sku: str) -> tuple[str, bool]:
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    try:
        async with session.head(url, timeout=ClientTimeout(total=5)) as response:
            return sku, response.status != 200  # True = mancante
    except Exception:
        return sku, True  # In caso di errore, considera mancante

async def controlla_foto_exist(sheet_id: str):
    sheet_lista = get_sheet(sheet_id, "LISTA")
    all_rows = sheet_lista.get_all_values()

    if len(all_rows) < 3:
        return

    header = all_rows[1]
    data = all_rows[2:]

    # Trova indice delle colonne
    col_sku = header.index("SKU") if "SKU" in header else 0
    col_scattare = header.index("SCATTARE") if "SCATTARE" in header else 10

    # Filtra solo righe con SKU e SCATTARE == True
    rows_to_check = [row for row in data if len(row) > col_scattare and row[col_sku] and row[col_scattare].strip().lower() == "true"]
    sku_list = [row[col_sku] for row in rows_to_check]

    results = {}

    connector = aiohttp.TCPConnector(limit=100)  # massimo 100 connessioni simultanee
    timeout = ClientTimeout(total=10)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [check_single_photo(session, sku) for sku in sku_list]
        responses = await asyncio.gather(*tasks, return_exceptions=False)

    for sku, missing in responses:
        results[sku] = missing

    # Costruisci nuova colonna K
    valori_k = []
    for row in data:
        sku = row[col_sku] if len(row) > col_sku else ""
        val = str(results.get(sku, row[col_scattare].strip().lower() == "true"))  # mantieni valore se non ricontrollato
        valori_k.append([val])

    # Scrivi in K3:K...
    range_k = f"K3:K{len(valori_k) + 2}"
    sheet_lista.update(values=valori_k, range_name=range_k, value_input_option="RAW")

@st.cache_data(ttl=300)
def carica_lista_foto(sheet_id: str, cache_key: str = "") -> pd.DataFrame:
    try:
        sheet = get_sheet(sheet_id, "LISTA")
        values = sheet.get("A3:O5000")
        if not values:
            return pd.DataFrame()

        # ✅ Definizione corretta: 11 intestazioni per colonne A–K
        headers = ["SKU", "CANALE", "STAGIONE", "COLLEZIONE", "DESCRIZIONE", "ALT1", "ALT2", "ALT3", "TGCAMP", "TGDISP", "SCATTARE", "ALT6", "RISCATTARE", "DISPONIBILITE", "CONSEGNATA"]
        df = pd.DataFrame(values, columns=headers)
        df = df[df["SKU"].notna() & (df["SKU"].str.strip() != "")]

        # 🧹 Normalizza booleani
        #df["SCATTARE"] = df["SCATTARE"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        #df["SCATTARE"] = df["SCATTARE"].fillna(False)
        #df["CONSEGNATA"] = df["CONSEGNATA"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        #df["CONSEGNATA"] = df["CONSEGNATA"].fillna(False)
        #df["RISCATTARE"] = df["RISCATTARE"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
        #df["RISCATTARE"] = df["RISCATTARE"].fillna(False)

        def normalize_bool(col):
            return col.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)
        
        df["SCATTARE"] = normalize_bool(df["SCATTARE"])
        df["CONSEGNATA"] = normalize_bool(df["CONSEGNATA"])
        df["RISCATTARE"] = normalize_bool(df["RISCATTARE"])

        return df[["SKU", "STAGIONE", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE", "RISCATTARE", "CONSEGNATA"]]
    except Exception as e:
        st.error(f"Errore durante il caricamento: {str(e)}")
        return pd.DataFrame()

# ---------------------------
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
# st.title("👟 Generatore Descrizioni di Scarpe con RAG")

# 📁 Caricamento dati
# Sidebar: menu
with st.sidebar:
    DEBUG = st.checkbox("🪛 Debug")
    st.markdown("## 📋 Menu")
    page = st.radio(
    "Seleziona sezione",  # label visibile a lettori di schermo
    ["🏠 Home", "📝 Descrizioni", "📸 Foto"],
    label_visibility="collapsed"  # nasconde visivamente la label
)

# ---------------------------
# 🏠 HOME
# ---------------------------
if page == "🏠 Home":
    st.subheader("📌 Benvenuto")
    st.markdown("""
    Benvenuto nell'app di generazione descrizioni prodotto per calzature.  
    Utilizza il menu a sinistra per iniziare.
    
    Funzionalità disponibili:
    - Generazione asincrona di descrizioni multilingua
    - Supporto RAG con FAISS
    - Captioning automatico immagine (BLIP)
    - Logging su Google Sheets
    - Salvataggio ZIP con CSV multilingua
    """)

# ---------------------------
# 📝 GENERAZIONE DESCRIZIONI
# ---------------------------
elif page == "📝 Descrizioni":
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
    if st.button("🚀 Genera Descrizioni"):
        st.session_state["generate"] = True
    
    if st.session_state.get("generate"):
        from io import BytesIO
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

elif page == "📸 Foto":
    st.header("📸 Gestione Foto")
    tab_names = ["ECOM", "ZFS", "AMAZON"]
    sheet_id = st.secrets["FOTO_GSHEET_ID"]

    col1, col2, spacer1, spacer2, col3 = st.columns(5)
    with col1:
        if st.button("📦 Genera lista SKU"):
            try:
                genera_lista_sku(sheet_id, tab_names)
                st.toast("✅ Lista SKU aggiornata!")
            except Exception as e:
                st.error(f"Errore: {str(e)}")
    with col2:
        if st.button("🔍 Controlla esistenza foto"):
            try:
                with st.spinner("🔍 Controllo asincrono delle foto..."):
                    asyncio.run(controlla_foto_exist(sheet_id))
                st.session_state["refresh_foto_token"] = str(time.time())
                st.toast("✅ Controllo foto completato!")
            except Exception as e:
                st.error(f"Errore durante il controllo: {str(e)}")
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state["refresh_foto_token"] = str(time.time())
    
    # 🔽 Caricamento dati con chiave cache dinamica
    cache_token = str(st.session_state.get("refresh_foto_token", "static"))
    df = carica_lista_foto(sheet_id, cache_key=cache_token)
    st.session_state["df_lista_foto"] = df

    # 📊 Riepilogo
    total = len(df)
    consegnate = df["CONSEGNATA"].sum()
    da_scattare = df["SCATTARE"].sum()
    scattate = total - da_scattare
        
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Totale SKU", total)
    c2.metric("🚚 Dal fotografo", consegnate)
    c3.metric("📸 Da scattare", da_scattare)
    c4.metric("✅ Già scattate", scattate)
   

    # 🔽 Filtro visualizzazione
    filtro_foto = st.selectbox("📌 Filtro foto da fare", ["Tutti", "Solo da scattare", "Solo già scattate"])

    if df.empty:
        st.warning("Nessuna SKU disponibile.")
    else:
        # 🔍 Applica filtro
        if filtro_foto == "Solo da scattare":
            df = df[df["SCATTARE"] == True]
        elif filtro_foto == "Solo già scattate":
            df = df[df["SCATTARE"] == False]

        # ✅ Visualizzazione con emoji
        def format_checkbox(val):
            if val is True:
                return "✅"
            elif val is False:
                return "❌"
            return "⛔️"

        #df_vista = df.copy()
        #df_vista = df_vista[["SKU", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE"]]
        #df_vista["Foto da fare"] = df_vista["SCATTARE"].apply(format_checkbox)
        #df_vista = df_vista.drop(columns=["SCATTARE"])

        # ✅ Rinomina colonne
        #df_vista.columns = ["SKU", "CANALE", "COLLEZIONE", "DESCRIZIONE", "📷"]

        # ✅ Costruzione vista tabellare con emoji
        df_vista = df.copy()
        df_vista = df_vista[["SKU", "CANALE", "COLLEZIONE", "DESCRIZIONE", "SCATTARE", "RISCATTARE"]]
        
        df_vista["📷"] = df_vista["SCATTARE"].apply(format_checkbox)
        df_vista["🔁"] = df_vista["RISCATTARE"].apply(format_checkbox)
        
        # Rimuovi le colonne booleane originali
        df_vista = df_vista.drop(columns=["SCATTARE", "RISCATTARE"])
        
        # ✅ Visualizzazione
        st.dataframe(df_vista, use_container_width=True)

    # Foto da riscattare
    st.subheader("🔁 Ristampa foto specifica")
    # ✅ Considera solo SKU che hanno già la foto (SCATTARE == False)
    df_foto_esistenti = df[df["SCATTARE"] == False]
    
    sku_input = st.text_input("🔍 Inserisci SKU da cercare (solo con foto esistenti)")
    
    selected_ristampe = st.session_state.get("ristampe_selezionate", set())
    
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
                ristampa_checkbox = st.checkbox("🔁 Ristampa", key=f"ristampa_{row['SKU']}")
                if ristampa_checkbox:
                    selected_ristampe.add(row['SKU'])
                else:
                    selected_ristampe.discard(row['SKU'])
    
    st.session_state["ristampe_selezionate"] = selected_ristampe
    
    if selected_ristampe:
        st.markdown(f"📦 SKU selezionate per ristampa: `{', '.join(sorted(selected_ristampe))}`")
        if st.button("✅ Conferma selezione per ristampa"):
            try:
                sheet = get_sheet(sheet_id, "LISTA")
                all_rows = sheet.get_all_values()
                headers = all_rows[1]
                data_rows = all_rows[2:]
    
                col_sku = 0
                col_ristampare = 13  # colonna N
    
                nuovi_valori = []
                for row in data_rows:
                    sku = row[col_sku].strip()
                    val = "TRUE" if sku in selected_ristampe else row[col_ristampare] if len(row) > col_ristampare else ""
                    nuovi_valori.append([val])
    
                range_update = f"N3:N{len(nuovi_valori) + 2}"
                sheet.update(values=nuovi_valori, range_name=range_update)
                st.success("✅ Ristampe aggiornate correttamente!")
                st.session_state["ristampe_selezionate"] = set()
            except Exception as e:
                st.error(f"❌ Errore aggiornamento: {str(e)}")
