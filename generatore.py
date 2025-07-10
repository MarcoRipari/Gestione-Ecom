import streamlit as st
import pandas as pd
import re
import json
import tempfile
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from datetime import datetime
import io
import zipfile

# === CONFIG ===
OPENAI_API_KEY = "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"
SHEET_ID = "1R9diynXeS4zTzKnjz1Kp1Uk7MNJX8mlHgV82NBvjXZc"

CREDENTIALS_JSON = {
  "type": "service_account",
  "project_id": "generazione-descrizioni-marco",
  "private_key_id": "e28bf7469e35b80d166c6fad3a691d0fd879ed85",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC0jT9HTYVpqug1\nXtt92iDvfpDkFrEGz1hd1Ctdqq2f8/aiswZFAsvmRf9bkOtfJgsEjdoIBl6D3qze\nUh9by4jdFAJIt2ox/u9Gn5gUttQqJk9iTmOoQNJLMqS1AV33R/IVi2UDu95RrWKk\nYE5ZomgXYIdcT31q8ufzZl18TuZCj2FNdZ44aR22+2emN3GP5HWz9vCwa5wsaCC8\nwRo7uzzeOlNNMGsaYYAhFwK3KSUVmJvg6C9sxDNfnjvoThbzcvnO3429Wlkei4cO\nebES5Zu/K27ieHcq0cFXeVhO6fB8vjsn4w6u1jDF8K/dKld0p5iakYqHPc+cbr2G\nonyqdEXnAgMBAAECggEAHgnn7r17hk6MbqC3BNO/KglVItWRo0/o5Edx5ZYJZ/TH\nYl9FmkKyWL/pkbrlJgHm0F2nWjFxFSB9g0mHdRbCUQHMtXtqfCHtfkL8IuoeF1sj\nVvgyxWHvetpUo9az4vnB0YrNBheCD/W4VR++uVP3XHhPXPDOrXX3WDv+LrnTvlvj\nF572cL7UlDRA6DEpjs5ObeSRvqgY6N+zFPr7Ze+8P7m8n8KhZdUjqwUdQICmGKtI\npH6venVsv24FgF0OqRP8NVHm6te7ZYk8bmPvlKrSXtYjDupcO+o5tXv1br613CpY\nheqiRr/tJ/JMibMxxRLTGAx3z8MawOWK0mi7Vf+AxQKBgQD6P0r+Q4YJ0mcK9Z44\ny2lxaTwI/aY/Z5z4QHT9BZOmJHsGgnoBuz30YFb1jgrsPDCZl7eVQLB82ll+6dQG\nkq7WK9lUk8BsC8eKYPYa13fJWH8uXYvJNz30i3YUIQN/M1tJxh1e693pWzD9yH/x\nqRkgvF1PTIgDIfOSGwzLh6WL0wKBgQC4s8uy8is7XBoPFQ/vmM01zcbINUTurYML\nWyHU8i5Am0u0xHAQ0pUuzAltfbDPGv8xVKbIa/0BFKKGibRBX/bnuUePFruNvlVD\nImfk9y2/ZC8W7OzjJdKdRWfK1issiD7yFIMJqg3ernZiPc58MAIGxikPDHuJNkG/\nUFAeaYR1HQKBgAC6tH4/NiHLMi+u/ZIOzbTd6KXiD1z58VQr4+tk28RNMOqY8MAW\nipyutzIqAtAjcMTR02Ak+x6yCDa9ebe3L7lCEXUUpSfrdN5rX+w+GoREtMIu1Zx1\ng8G1sldmrTrurGJvqGBBcbkfYeorbmwG4SLeSatUfsT7kVkoqQXi1FGvAoGAIyK4\nxkLBLJqZrnLQREDqEKkjfmR7x3ekbR2Z8vtbBxlDrpCLzPdyP6O6y2RUpSE6mHTF\nAW1hhLobLMK3UpRh0LTzQuoNJaqmZ438+5Z10mnJd2/8pD1GsnpIg1J4hhEpAD4c\nq1L5Lno7tPaS+Bbd29IIb39tZK24lh8+Dnr+IpUCgYEA7DFxcu4eafrUXNwyhnzW\nu5todlwzwU0Ev4z4BnBt8Kp+n9EM2hyHrP9VH/Fzy3tmZ0Ak3qEErvf4/ZVgRKK6\nvjsnrTOmSQcwJ0JoH1DUl2CSKSxN9TkbgOtU37bseTlvQhwxq3o0LCb3k4c4x+My\nfOh/boG80Q1j71qtB5G8U3w=\n-----END PRIVATE KEY-----\n",
  "client_email": "marcoripari@generazione-descrizioni-marco.iam.gserviceaccount.com",
  "client_id": "110026165625284721257",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/marcoripari%40generazione-descrizioni-marco.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

LINGUE_TARGET = ["it", "en", "fr", "de"]  # Italiano, Inglese, Francese, Tedesco

# === SETUP ===
st.set_page_config(page_title="Generatore Descrizioni", layout="centered")
st.title("📝 Generatore Descrizioni Prodotto")

# === GOOGLE SHEET ===
def connect_to_gsheet(credentials_dict, sheet_id):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        json.dump(credentials_dict, temp)
        temp.flush()
        creds = ServiceAccountCredentials.from_json_keyfile_name(temp.name, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    return sheet

def log_audit(sheet, action, sku, status, message):
    try:
        try:
            worksheet = sheet.worksheet("AuditTrail")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sheet.add_worksheet(title="AuditTrail", rows="100", cols="5")
            worksheet.append_row(["Timestamp", "Azione", "SKU", "Stato", "Dettagli"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, action, sku, status, message]
        worksheet.append_row(row)

    except Exception as e:
        st.warning(f"⚠️ Impossibile scrivere nell'AuditTrail: {e}")

# === CACHING & RAG BASE ===
cached_data = None
similarity_index = None
vectorizer = TfidfVectorizer()

@st.cache_data(show_spinner=False)
def load_gsheet_cache():
    try:
        sheet = connect_to_gsheet(CREDENTIALS_JSON, SHEET_ID)
        data = sheet.sheet1.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"⚠️ Errore nel caricamento da Google Sheet: {e}")
        return pd.DataFrame()  # meglio che None

cached_data = load_gsheet_cache()
if cached_data.empty or "SKU" not in cached_data.columns:
    st.warning("⚠️ Il foglio Google Sheet è vuoto o non contiene la colonna 'SKU'.")
else:
    st.write("✅ Colonne disponibili:", cached_data.columns.tolist())


# === COSTRUISCI FAISS-LIKE INDEX ===
def build_similarity_index(df):
    if df.empty:
        return None, None
    corpus = []
    skus = []
    for _, row in df.iterrows():
        sku = str(row.get("SKU", ""))[:7]  # modello base
        text = " ".join([str(v) for k, v in row.items() if k.lower() not in ["description", "short_description"]])
        corpus.append(text)
        skus.append(sku)
    X = vectorizer.fit_transform(corpus)
    return X, skus

X_cache, skus_cache = build_similarity_index(cached_data)

# === FUNZIONE OPENAI ===
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_descriptions(row):
    sku_prefix = str(row.get("SKU", ""))[:7]
    product_info = ", ".join([
        f"{k}: {v}" for k, v in row.items()
        if k.lower() not in ["description", "short_description"] and pd.notna(v)
    ])

    # === RAG MULTI-LIVELLO ===
    examples = []
    if cached_data is not None and not cached_data.empty:
        matches = cached_data[cached_data["SKU"].astype(str).str.startswith(sku_prefix)]

        # Livello 2: Similarità semantica
        if X_cache is not None:
            row_text = " ".join([str(v) for k, v in row.items() if k.lower() not in ["description", "short_description"]])
            X_row = vectorizer.transform([row_text])
            sims = cosine_similarity(X_row, X_cache)[0]
            top_indices = sims.argsort()[-3:][::-1]  # top 3 simili
            similar_rows = cached_data.iloc[top_indices]
            matches = pd.concat([matches, similar_rows]).drop_duplicates(subset="SKU")

        # Livello 3: Matching per categoria
        if "Categoria" in row and "Categoria" in cached_data.columns:
            matches = pd.concat([matches, cached_data[cached_data["Categoria"] == row["Categoria"]]])
            matches = matches.drop_duplicates(subset="SKU")

        for _, ex in matches.iterrows():
            examples.append(f"- {ex.get('description', '')}\n  ➜ {ex.get('short_description', '')}")

    # === Costruzione Prompt ===
    context = "\n".join(examples[:5]) if examples else ""
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

Esempi:
{context}

Scrivi solo un oggetto JSON con questo formato esatto (niente testo extra):

{{
  "description": "...",  // circa 60 parole
  "short_description": "..."   // circa 20 parole
}}

Dettagli del prodotto: {product_info}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        long_desc = result.get("description", "Descrizione lunga non trovata")
        short_desc = result.get("short_description", long_desc[:100])
        return long_desc, short_desc
    except:
        return "Errore", "Errore"

# === FUNZIONE TRADUZIONE ===

def translate_descriptions(long_desc, short_desc, target_lang):
    prompt = f"""
Traduci il seguente oggetto JSON in lingua '{target_lang}' mantenendo uno stile:
- accattivante
- caldo
- professionale
- user friendly
- SEO friendly

JSON:
{{
  "description": "...",
  "short_description": "..."
}}

Rispondi solo con l'oggetto JSON tradotto, senza testo extra.
"""
  
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get("description", ""), result.get("short_description", "")
    except:
        return "Errore traduzione", "Errore traduzione"

# === APP STREAMLIT ===

st.title("📝 Generatore descrizioni calzature")

uploaded_file = st.file_uploader("📤 Carica il file CSV con colonna 'SKU'", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "SKU" not in df.columns:
        st.error("❌ Il file CSV deve contenere una colonna 'SKU'.")
    else:
        results_per_lang = defaultdict(list)
        sheet = connect_to_gsheet(CREDENTIALS_JSON, SHEET_ID)

        # Caching foglio Google
        try:
            existing_sheet = sheet.worksheet("IT")
            cached_data = pd.DataFrame(existing_sheet.get_all_records())
        except:
            cached_data = None

        # Stima costi prima della generazione
        if st.button("📊 Stima costo descrizioni"):
          num_prodotti = len(df)
          tokens_per_desc = 150  # stima base: descrizione lunga + corta
          lingue_extra = [lang for lang in LINGUE_TARGET if lang != "it"]
          tokens_per_traduzione = 120  # stima per ciascuna traduzione

          total_tokens = num_prodotti * (tokens_per_desc + len(lingue_extra) * tokens_per_traduzione)
          cost = (total_tokens / 1000) * 0.0015

          st.info(f"🧮 Totale prodotti: {num_prodotti}")
          st.info(f"🌍 Traduzioni attive: {', '.join(lingue_extra) if lingue_extra else 'nessuna'}")
          st.info(f"🔢 Token stimati: ~{total_tokens}")
          st.success(f"💰 Costo stimato: ~{cost:.4f} USD")

        if st.button("🚀 Conferma e genera"):
            st.info("🔄 Generazione in corso...")
            progress = st.progress(0)
            results = []

            for idx, row in df.iterrows():
                sku = str(row.get("SKU", ""))
                if cached_data is not None and "SKU" in cached_data.columns and sku in cached_data["SKU"].astype(str).values:
                    long_desc = cached_data[cached_data["SKU"] == sku]["description"].values[0]
                    short_desc = cached_data[cached_data["SKU"] == sku]["short_description"].values[0]
                else:
                    long_desc, short_desc = generate_descriptions(row)
                    if long_desc == "Errore":
                        log_audit(sheet, action="generate", sku=sku, status="failed", message="Errore durante generazione")
                    else:
                        log_audit(sheet, action="generate", sku=sku, status="success", message="Descrizione generata")

                for lang in LINGUE_TARGET:
                    if lang == "it":
                        desc, short = long_desc, short_desc
                    else:
                        desc, short = translate_descriptions(long_desc, short_desc, lang)

                    new_row = row.copy()
                    new_row["description"] = desc
                    new_row["short_description"] = short
                    results_per_lang[lang].append(new_row)

                progress.progress((idx + 1) / len(df))

                # === SALVATAGGIO SU GOOGLE SHEETS E ZIP ===
                zip_buffer = io.BytesIO()
    
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for lang, rows in results_per_lang.items():
                        lang_df = pd.DataFrame(rows).replace({np.nan: None})
                        lang_sheet_name = lang.upper()
    
                        try:
                            worksheet = None
                            try:
                                worksheet = sheet.worksheet(lang_sheet_name)
                            except gspread.exceptions.WorksheetNotFound:
                                worksheet = sheet.add_worksheet(title=lang_sheet_name, rows="100", cols="20")
                                worksheet.append_row(lang_df.columns.tolist())
    
                            existing = worksheet.get_all_values()
                            existing_skus = set(row[0] for row in existing[1:]) if len(existing) > 1 else set()
                            new_rows = []
    
                            for _, row in lang_df.iterrows():
                                sku = str(row["SKU"])
                                if sku not in existing_skus:
                                    new_rows.append(list(row.values))
                                    log_audit(sheet, "Generazione", sku, "Successo", f"Aggiunto a {lang_sheet_name}")
                                else:
                                    log_audit(sheet, "Generazione", sku, "Ignorato", f"SKU già presente in {lang_sheet_name}")
    
                            if new_rows:
                                worksheet.append_rows(new_rows)
    
                        except Exception as e:
                            st.error(f"❌ Errore salvataggio per lingua {lang}: {e}")
    
                        # Scrivi CSV nello ZIP
                        csv_bytes = lang_df.to_csv(index=False).encode("utf-8")
                        zip_file.writestr(f"descrizioni_{lang_sheet_name}.csv", csv_bytes)
    
                zip_buffer.seek(0)
    
                # ✅ Mostra solo dopo la generazione
                st.download_button(
                    label="⬇️ Scarica tutti i CSV in ZIP",
                    data=zip_buffer,
                    file_name="descrizioni_multilingua.zip",
                    mime="application/zip"
                )
