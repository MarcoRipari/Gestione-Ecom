import streamlit as st
import pandas as pd
import openai
import zipfile
import io
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import faiss
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from typing import List, Dict

# CONFIGURAZIONI ------------------------
openai.api_key = "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"
GOOGLE_SHEET_ID = "1R9diynXeS4zTzKnjz1Kp1Uk7MNJX8mlHgV82NBvjXZc"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
GOOGLE_CREDENTIALS = {
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

LANGUAGES = ["it", "en", "fr", "de"]
DEFAULT_LANG = "it"
COLUMN_WEIGHTS = {
    "SKU famille": 1.0,
    "Subtitle": 0.8, # Tipo di modello
    "Concept": 0.7, # Concetto usabilità
    "Sp.feature": 0.6, # Features
    "Sexe": 0.5, # Sesso
    "Saison": 0.4, # Stagione
    "sole_material_zalando": 0.4, # Soletta interna
    "shoe_fastener_zalando": 0.2, # Chiusura
    "futter_zalando": 0.5 # Fodera
    # aggiungi altre colonne qui
}

# Funzioni ---------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def authenticate_google():
    creds = ServiceAccountCredentials.from_json_keyfile_dict(GOOGLE_CREDENTIALS, SCOPES)
    return gspread.authorize(creds)

def read_sheet(sheet_name: str):
    client = authenticate_google()
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def append_to_sheet(sheet_name: str, df: pd.DataFrame):
    client = authenticate_google()
    sheet = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws = sheet.worksheet(sheet_name)
    except:
        ws = sheet.add_worksheet(title=sheet_name, rows=10000, cols=50)
    ws.append_rows(df.values.tolist(), value_input_option='RAW')

def build_prompt(row: pd.Series, examples: List[str]) -> str:
    base = "Scrivi due descrizioni per un prodotto calzatura da vendere online:\n"
    base += "- Descrizione lunga (~60 parole)\n"
    base += "- Descrizione corta (~20 parole)\n\n"
    base += "Mantieni questo stile:\n"
    base += "- accattivante\n"
    base += "- caldo\n"
    base += "- professionale\n"
    base += "- user friendly\n"
    base += "- SEO friendly\n\n"
    base += "Questi sono i campi più rilevanti:\n"
    base += "- 'Subtitle' = tipo di modello\n"
    base += "- 'Concept' = Concetto usabilità\n"
    base += "- 'Sp.feature' = qualità/pregi\n"
    base += "- 'Sexe' = sesso\n"
    base += "- 'Saison' = stagione\n"
    base += "- 'sole_material_zalando' = soletta interna\n"
    base += "- 'shoe_fastener_zalando' = tipo di chiusura\n"
    base += "- 'futter_zalando' = fodera\n\n"
    base += "Esempi simili:\n"
    base += "\n\n".join(examples[:3])
    base += f"\n\nDati prodotto:\n"
    for col, val in row.items():
        base += f"{col}: {val}\n"
    base += "\nOutput:"
    return base

def generate_with_openai(prompt: str) -> Dict[str, str]:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        text = response.choices[0].message.content
        lines = text.strip().split("\n")
        long_desc = next((l for l in lines if len(l.split()) > 8), "")
        short_desc = next((l for l in lines if 3 < len(l.split()) <= 15), "")
        return {"description": long_desc, "description2": short_desc, "raw": text}
    except Exception as e:
        return {"error": str(e)}

def build_faiss_index(df: pd.DataFrame, model) -> faiss.IndexFlatIP:
    vectors = []
    for _, row in df.iterrows():
        text = " ".join(str(row[col]) * int(COLUMN_WEIGHTS.get(col, 1) * 10) for col in df.columns)
        emb = model.encode([text])[0]
        vectors.append(emb)
    matrix = np.vstack(vectors)
    matrix = normalize(matrix, axis=1)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, vectors

def retrieve_similar(row, df, index, vectors, model, k=3):
    text = " ".join(str(row[col]) * int(COLUMN_WEIGHTS.get(col, 1) * 10) for col in df.columns)
    query = model.encode([text])[0]
    query = normalize([query])[0]
    D, I = index.search(np.array([query]), k)
    return [f"- {df.iloc[i]['description']}\n- {df.iloc[i]['description2']}" for i in I[0] if 'description' in df.columns]

def translate(text: str, target_lang: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Traduci in {target_lang.upper()} mantenendo la lunghezza e lo stile:\n{text}"
            }]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Errore traduzione: {e}]"

def estimate_tokens(df: pd.DataFrame, lang_list: List[str]) -> int:
    avg_prompt_length = 250  # token stimati per prompt
    multiplier = len(lang_list)
    return int(len(df) * avg_prompt_length * multiplier)

# UI ---------------------------------

st.title("🧠 Generatore Descrizioni Scarpe (Multilingua)")

lang_selection = st.multiselect("Seleziona le lingue", LANGUAGES, default=[DEFAULT_LANG])

uploaded_file = st.file_uploader("Carica CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df_out = df.copy()
    df_out["description"] = ""
    df_out["description2"] = ""

    st.write("Anteprima:")
    st.dataframe(df.head())

    if st.button("Stima costo"):
        total_tokens = estimate_tokens(df, lang_selection)
        total_cost = total_tokens / 1000 * 0.001  # costo gpt-3.5-turbo
        st.info(f"Token stimati: {total_tokens} | Costo stimato: ${total_cost:.4f}")

    if st.button("Genera descrizioni"):
        st.write("Caricamento storico...")
        storico_df = read_sheet(DEFAULT_LANG)
        model = load_model()
        index, vecs = build_faiss_index(storico_df, model)

        results = []
        logs = []

        for i, row in df.iterrows():
            try:
                examples = retrieve_similar(row, storico_df, index, vecs, model)
                prompt = build_prompt(row, examples)
                result = generate_with_openai(prompt)
                if "error" in result:
                    raise Exception(result["error"])
                row_out = row.to_dict()
                row_out.update({
                    "description": result["description"],
                    "description2": result["description2"]
                })

                for lang in lang_selection:
                    if lang != DEFAULT_LANG:
                        row_out[f"description_{lang}"] = translate(result["description"], lang)
                        row_out[f"description2_{lang}"] = translate(result["description2"], lang)

                df_out.loc[i, "description"] = result["description"]
                df_out.loc[i, "description2"] = result["description2"]

                for lang in lang_selection:
                    if lang != DEFAULT_LANG:
                        df_out.loc[i, f"description_{lang}"] = row_out[f"description_{lang}"]
                        df_out.loc[i, f"description2_{lang}"] = row_out[f"description2_{lang}"]

                logs.append([datetime.now().isoformat(), row["SKU"], "SUCCESS", prompt, result["raw"]])
                results.append(row_out)

            except Exception as e:
                logs.append([datetime.now().isoformat(), row["SKU"], "ERROR", str(e), ""])

        # Salvataggio
        for lang in lang_selection:
            lang_df = df_out[["SKU", f"description_{lang}" if lang != "it" else "description",
                              f"description2_{lang}" if lang != "it" else "description2"]].copy()
            append_to_sheet(lang, lang_df)

        # Log
        log_df = pd.DataFrame(logs, columns=["timestamp", "SKU", "status", "prompt", "output"])
        append_to_sheet("logs", log_df)

        # Zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for lang in lang_selection:
                lang_df = df_out[["SKU", f"description_{lang}" if lang != "it" else "description",
                                  f"description2_{lang}" if lang != "it" else "description2"]]
                csv_bytes = lang_df.to_csv(index=False).encode("utf-8")
                zip_file.writestr(f"descrizioni_{lang}.csv", csv_bytes)

        st.download_button("📥 Scarica CSV ZIP", zip_buffer.getvalue(), "descrizioni.zip", mime="application/zip")
