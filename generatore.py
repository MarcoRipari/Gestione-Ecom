import streamlit as st
import pandas as pd
import io
import os
from openai import OpenAI
import google.auth
from google.oauth2 import service_account
import gspread

# Modello da usare
MODEL = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = 0.0015  # stimato per gpt-3.5-turbo input/output

# Carica le credenziali di Google Sheets dal file
def connect_to_gsheet(json_path, sheet_id):
    creds = service_account.Credentials.from_service_account_file(
        json_path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    return sheet

# Generazione delle descrizioni
def generate_descriptions(row, api_key):
    client = OpenAI(api_key=api_key)

    product_info = ", ".join([
        f"{k}: {v}" for k, v in row.items()
        if k.lower() not in ["description", "short_description"] and pd.notna(v)
    ])

    prompt = f"""
Scrivi due descrizioni per una calzatura:
1. Descrizione lunga di circa 60 parole.
2. Descrizione breve di circa 20 parole.
Tono: accattivante, caldo, professionale, user-friendly e SEO-friendly.
Dettagli: {product_info}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    result = response.choices[0].message.content.strip()
    parts = [p.strip("1234567890.-: \n") for p in result.split("\n") if p.strip()]
    long_desc = parts[0] if len(parts) > 0 else "Descrizione non trovata"
    short_desc = parts[1] if len(parts) > 1 else long_desc[:100]

    return long_desc, short_desc

# App Streamlit
st.title("🥿 Generatore di Descrizioni per Calzature")

# API Key
api_key = st.text_input("🔑 Inserisci la tua OpenAI API Key", type="password")
if api_key:
    st.success("✅ API Key impostata correttamente.")

# Caricamento file CSV
uploaded_file = st.file_uploader("📁 Carica il file CSV con i prodotti", type="csv")

# Caricamento credenziali Google e Sheet ID
credentials_file = st.file_uploader("🔐 Carica il file credentials.json di Google", type="json")
sheet_id = st.text_input("📝 Inserisci lo Sheet ID del tuo Google Sheet")

if uploaded_file and credentials_file and sheet_id and api_key:
    df = pd.read_csv(uploaded_file)
    if 'description' not in df.columns or 'short_description' not in df.columns:
        st.error("❌ Il file deve contenere le colonne 'description' e 'short_description'.")
    else:
        # Stima token (approssimativa): 1 token ≈ 4 caratteri
        total_chars = df.drop(columns=["description", "short_description"]).astype(str).applymap(len).sum().sum()
        estimated_tokens = total_chars / 4
        estimated_cost = (estimated_tokens / 1000) * COST_PER_1K_TOKENS

        st.markdown(f"💰 **Stima costo generazione**: {estimated_tokens:.0f} token ≈ ${estimated_cost:.4f}")
        generate_now = st.button("✏️ Genera descrizioni ora")

        if generate_now:
            client = OpenAI(api_key=api_key)
            long_descs, short_descs = [], []

            progress = st.progress(0)
            for idx, row in df.iterrows():
                long, short = generate_descriptions(row, client)
                long_descs.append(long)
                short_descs.append(short)
                progress.progress((idx + 1) / len(df))

            df["description"] = long_descs
            df["short_description"] = short_descs
            st.success("✅ Descrizioni generate con successo!")

            # Salvataggio su Google Sheets
            gsheet = connect_to_gsheet(credentials_file.name, sheet_id)
            gsheet.clear()
            gsheet.update([df.columns.values.tolist()] + df.values.tolist())
            st.success("📤 Dati salvati su Google Sheets!")

            # Download CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            st.download_button("📥 Scarica il file con descrizioni", output.getvalue(), file_name="output.csv", mime="text/csv")
