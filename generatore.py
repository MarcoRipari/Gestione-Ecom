import streamlit as st
import pandas as pd
import openai
import gspread
from google.oauth2.service_account import Credentials
import json
import time

# Costanti
MODEL = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = 0.0015  # GPT-3.5 Turbo input/output price (July 2025)

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
st.title("👟 Generatore Descrizioni Prodotto (GPT-3.5)")

# === SIDEBAR ===
st.sidebar.header("🔐 Impostazioni")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
use_gsheet = st.sidebar.checkbox("Salva su Google Sheets")
sheet_id = st.sidebar.text_input("ID Google Sheet (facoltativo)", disabled=not use_gsheet)
json_file = st.sidebar.file_uploader("Credenziali Google (JSON)", type=["json"], disabled=not use_gsheet)

if api_key:
    openai.api_key = api_key
    st.sidebar.success("✅ API Key caricata")
else:
    st.sidebar.warning("⚠️ Inserisci la tua OpenAI API Key per continuare")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Carica un file CSV con i prodotti")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Anteprima del file caricato:")
    st.dataframe(df.head())

    # Stima token
    def estimate_token_cost(data):
        est_tokens = 0
        for _, row in data.iterrows():
            content = " ".join([str(v) for k, v in row.items() if k not in ["description", "short_description"]])
            est_tokens += len(content.split()) * 1.5  # stima approssimativa
        return int(est_tokens), round((est_tokens / 1000) * COST_PER_1K_TOKENS, 4)

    token_count, estimated_cost = estimate_token_cost(df)
    st.info(f"📈 Token stimati: **{token_count}** – Costo stimato: **${estimated_cost}**")

    confirm = st.button("✅ Conferma generazione descrizioni con OpenAI")

    # === GOOGLE SHEET ===
    def connect_to_gsheet(json_file, sheet_id):
        credentials = Credentials.from_service_account_info(json.load(json_file),
                                                             scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet

    def save_to_gsheet(dataframe, sheet):
        sheet.clear()
        sheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())

    # === GENERAZIONE ===
    def generate_descriptions(row):
        product_info = ", ".join([f"{k}: {v}" for k, v in row.items() if k not in ["description", "short_description"] and pd.notna(v)])
        prompt = f"""
Scrivi due descrizioni per un prodotto di calzature:
1. Una descrizione lunga di circa 60 parole.
2. Una descrizione breve di circa 20 parole.
Tono accattivante, caldo, professionale, user-friendly e SEO-friendly.
Dettagli prodotto: {product_info}
"""

        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            output = response.choices[0].message.content.strip()
            parts = [p.strip("1234567890.-: \n") for p in output.split("\n") if p.strip()]
            if len(parts) < 2:
                long_desc = parts[0]
                short_desc = parts[0][:100]
            else:
                long_desc = parts[0]
                short_desc = parts[1]
            used_tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
            return long_desc, short_desc, used_tokens
        except Exception as e:
            return "Errore descrizione", "Errore breve", 0

    # === AVVIA GENERAZIONE ===
    if confirm:
        st.success("🚀 Generazione in corso...")
        progress = st.progress(0)
        total = len(df)
        results = []

        for idx, row in df.iterrows():
            desc, short, used_tokens = generate_descriptions(row)
            row["description"] = desc
            row["short_description"] = short
            row["token_used"] = used_tokens
            results.append(row)
            progress.progress((idx + 1) / total)
            time.sleep(0.5)

        result_df = pd.DataFrame(results)
        st.dataframe(result_df[["description", "short_description"]].head())

        # Salva su Google Sheets se richiesto
        if use_gsheet and json_file and sheet_id:
            try:
                sheet = connect_to_gsheet(json_file, sheet_id)
                save_to_gsheet(result_df, sheet)
                st.success("✅ Storico salvato su Google Sheets")
            except Exception as e:
                st.error(f"Errore Google Sheets: {e}")

        # Download CSV finale
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Scarica CSV con descrizioni", csv_data, file_name="output_descrizioni.csv")
