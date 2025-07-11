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
    example_section = "\n".join([f"Esempio: {r['Description']}" for _, r in examples.iterrows()])
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
    #df_input = read_csv_auto_encoding(uploaded)
    df_input = pd.read_csv(uploaded)
    st.dataframe(df_input.head())
    col_weights = {}
    st.markdown("### ⚙️ Configura i pesi delle colonne (importanza nella similarità)")
    for col in df_input.columns:
        if col not in ["Description", "Description2"]:
            col_weights[col] = st.slider(f"Peso colonna: {col}", 0, 5, 1, step=0.1)
    st.markdown("### 🔍 Anteprima descrizione per una riga")
    row_index = st.number_input("Indice della riga da testare (0-based)", min_value=0, max_value=len(df_input)-1, value=0)

    # Stimo il costo del token con RAG
    if st.button("📄 Stima costi con RAG"):
        # Recupera esempi se RAG è attivo
        try:
            if sheet_id:
                test_row = 0
                data_sheet = get_sheet(sheet_id, "it")
                df_storico = pd.DataFrame(data_sheet.get_all_records())
                index, index_df = build_faiss_index(df_storico, col_weights)
                simili = retrieve_similar(test_row, index_df, index, k=1, col_weights=col_weights)
        except Exception as e:
            st.warning(f"Errore simili: {e}")

        # Visualizza simili
        with st.expander("📋 FAISS generato"):
            st.code(index)
            st.code(index_df)
        with st.expander("📋 Simili recuperati"):
            st.code(simili)
            
    if st.button("Stima costi"):
        # Calcolo prompt medio sui primi 3 record
        prompts = []
        for _, row in df_input.iterrows():
            simili = pd.DataFrame([])  # niente RAG per la stima
            prompt = build_prompt(row, simili)
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
                    simili = retrieve_similar(row, index_df, index, k=1, col_weights=col_weights)
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
                df_export["Descrizione lunga"] = df_out.get("Description", "")
                df_export["Descrizione corta"] = df_out.get("Description2", "")

                csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                zf.writestr(f"descrizioni_{lang}.csv", csv_bytes)
        mem_zip.seek(0)
        st.success("✅ Generazione completata con successo!")
        st.download_button("📥 Scarica CSV (ZIP)", mem_zip, file_name="descrizioni.zip")
