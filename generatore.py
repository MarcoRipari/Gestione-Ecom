import streamlit as st
import pandas as pd
import openai
import os
import json
import tempfile
from google.oauth2.service_account import Credentials
import gspread

# Impostazioni
MODEL = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = 0.001  # solo input

def approx_token_count(text):
    return max(1, int(len(text) / 4))

def generate_descriptions(row):
    product_info = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
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
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()

        # Cerca di dividere il risultato
        parts = [p.strip("1234567890.-: \n") for p in output.split("\n") if p.strip()]
        if len(parts) < 2:
            long_desc = parts[0]
            short_desc = parts[0][:100]  # fallback breve
        else:
            long_desc = parts[0]
            short_desc = parts[1]

        used_tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        return long_desc, short_desc, used_tokens

    except Exception as e:
        return "Errore generazione descrizione", "Errore breve", 0

def connect_to_gsheet(credentials_file, sheet_id):
    creds = Credentials.from_service_account_file(credentials_file)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    return sheet

# Inizializzazione stato
if "stage" not in st.session_state:
    st.session_state.stage = 0
    st.session_state.total_tokens = 0
    st.session_state.costo_stimato = 0
    st.session_state.df = None

st.title("🥿 Generatore Descrizioni Calzature con GPT-3.5")

# API Key e file
openai.api_key = st.text_input("🔑 Inserisci la tua API Key OpenAI", type="password")
credentials_file = st.file_uploader("📄 Carica il file credentials.json per Google Sheets", type=["json"])
sheet_id = st.text_input("📝 Inserisci il Google Sheet ID")
csv_file = st.file_uploader("📦 Carica il file CSV dei prodotti", type=["csv"])

if openai.api_key:
    st.success("✅ API Key impostata correttamente!")

if st.button("🔄 Reset app"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Caricamento CSV
if csv_file:
    df = pd.read_csv(csv_file)
    st.session_state.df = df
    st.write("### 📋 Anteprima del file CSV")
    st.dataframe(df.head())

# Stima token
if st.session_state.df is not None and st.session_state.stage == 0:
    if st.button("💰 Stima costo generazione"):
        total_tokens = 0
        for _, row in st.session_state.df.iterrows():
            product_info = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            prompt = f"Scrivi due descrizioni per un prodotto: {product_info}"
            total_tokens += approx_token_count(prompt)
        cost = (total_tokens / 1000) * COST_PER_1K_TOKENS
        st.session_state.total_tokens = total_tokens
        st.session_state.costo_stimato = cost
        st.session_state.stage = 1
        st.success("✅ Stima completata")

# Mostra stima
if st.session_state.stage >= 1:
    st.info(f"🔢 Token stimati: {st.session_state.total_tokens}")
    st.warning(f"💲 Costo stimato: circa ${st.session_state.costo_stimato:.4f}")

# Conferma e generazione descrizioni
if st.session_state.stage == 1:
    if st.button("🚀 Conferma e Genera descrizioni"):
        long_descs, short_descs, token_counts = [], [], []
        for _, row in st.session_state.df.iterrows():
            long, short, tokens = generate_descriptions(row)
            long_descs.append(long)
            short_descs.append(short)
            token_counts.append(tokens)

        st.session_state.df["description"] = long_descs
        st.session_state.df["short_description"] = short_descs
        st.session_state.df["tokens"] = token_counts
        st.session_state.stage = 2
        st.success("✅ Descrizioni generate!")

# Mostra risultati
if st.session_state.stage >= 2:
    st.write("### 📄 Descrizioni generate")
    st.dataframe(st.session_state.df[["description", "short_description"]])

    if st.button("📤 Salva su Google Sheets"):
        if credentials_file and sheet_id:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(credentials_file.getvalue())
                tmp_path = tmp.name

            sheet = connect_to_gsheet(tmp_path, sheet_id)
            sheet.clear()
            sheet.update([st.session_state.df.columns.values.tolist()] + st.session_state.df.values.tolist())
            st.success("✅ Dati salvati su Google Sheets!")
        else:
            st.error("⚠️ Devi fornire il file credentials.json e il Google Sheet ID")
