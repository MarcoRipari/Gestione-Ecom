# generatore.py

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
def embed_texts(texts: List[str], model="text-embedding-3-small") -> List[List[float]]:
    response = openai.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]

def build_faiss_index(df: pd.DataFrame, col_weights: Dict[str, float], cache_path="faiss_cache.index"):
    if os.path.exists(cache_path):
        index = faiss.read_index(cache_path)
        return index, df

    texts = []
    for _, row in df.iterrows():
        text = " ".join([f"{col}: {row[col]}" * int(col_weights.get(col, 1)) for col in df.columns if pd.notna(row[col])])
        texts.append(text)

    vectors = embed_texts(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, cache_path)
    return index, df

def retrieve_similar(query_row: pd.Series, df: pd.DataFrame, index, k=5, col_weights: Dict[str, float] = {}):
    query_text = " ".join([f"{col}: {query_row[col]}" * int(col_weights.get(col, 1)) for col in df.columns if pd.notna(query_row[col])])
    query_vector = embed_texts([query_text])[0]
    D, I = index.search(np.array([query_vector]).astype("float32"), k)
    return df.iloc[I[0]]

# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_prompt(row, examples):
    example_section = "\n".join([f"Esempio: {r['description']}" for _, r in examples.iterrows()])
    prompt = f"""
Scrivi DUE descrizioni per una calzatura da vendere online, mantenendo uno stile:
- accattivante
- caldo
- professionale
- user friendly
- SEO friendly
Indicazioni:
- Evita di inserire nome del prodotto e marchio.
- Evita di inserire il colore
- Usa questi esempi per apprendere tono/stile

Esempi simili:
{example_section}

Informazioni prodotto:
{row.to_json()}

Usa questo output
Descrizione lunga:
Descrizione breve:
"""
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

# ---------------------------
# 📦 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni di Scarpe con RAG")

sheet_id = st.text_input("Google Sheet ID dello storico", key="sheet")
selected_langs = st.multiselect("Seleziona lingue di output", ["it", "en", "fr", "de"], default=["it"])
uploaded = st.file_uploader("Carica il CSV dei prodotti", type="csv")

# Configurazione pesi colonne per RAG
if uploaded:
    df_input = pd.read_csv(uploaded)
    col_weights = {}
    st.markdown("### ⚙️ Configura i pesi delle colonne (importanza nella similarità)")
    for col in df_input.columns:
        if col not in ["description", "description2"]:
            col_weights[col] = st.slider(f"Peso colonna: {col}", 0, 5, 1)

    if st.button("Stima costi"):
        est_tokens = len(df_input) * 250 * len(selected_langs)
        est_cost = est_tokens / 1000 * 0.001  # gpt-3.5-turbo
        st.info(f"Token stimati: {est_tokens} | Costo stimato: ${est_cost:.4f}")

    if st.button("Genera Descrizioni"):
        index_df = None
        if sheet_id:
            try:
                data_sheet = get_sheet(sheet_id, "it")
                df_storico = pd.DataFrame(data_sheet.get_all_records())
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
                    simili = retrieve_similar(row, index_df, index, k=3, col_weights=col_weights)
                else:
                    simili = pd.DataFrame([])

                prompt = build_prompt(row, simili)
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
                    #"description": descr_lunga.strip().replace("Descrizione lunga:", "").strip(),
                    #"description2": descr_breve.strip()
                    "description": descr_lunga,
                    "description2": descr_breve
                }

                for lang in selected_langs:
                    if lang == "it":
                        all_outputs[lang].append(base)
                    else:
                        trad_lunga = translate_text(base["description"], target_lang=lang)
                        trad_breve = translate_text(base["description2"], target_lang=lang)
                        trad = base.copy()
                        trad["description"] = trad_lunga
                        trad["description2"] = trad_breve
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
            save_to_sheet(sheet_id, lang, df_out)

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
                df_export["Descrizione lunga"] = df_out.get("description", "")
                df_export["Descrizione corta"] = df_out.get("description2", "")

                csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                zf.writestr(f"descrizioni_{lang}.csv", csv_bytes)
        mem_zip.seek(0)
        st.success("✅ Generazione completata con successo!")
        st.download_button("📥 Scarica CSV (ZIP)", mem_zip, file_name="descrizioni.zip")
