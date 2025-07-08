import streamlit as st
import pandas as pd
import openai
import gspread
from google.oauth2.service_account import Credentials
import json
import time

# Configurazione
MODEL = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = 0.0015

st.set_page_config(page_title="📝 Generatore Descrizioni Prodotto", layout="wide")
st.title("👟 Generatore Descrizioni Calzature (con OpenAI)")

# Sidebar: API e Sheets
st.sidebar.header("🔐 Impostazioni")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
use_gsheet = st.sidebar.checkbox("Salva su Google Sheets")
sheet_id = st.sidebar.text_input("ID Google Sheet", disabled=not use_gsheet)
json_file = st.sidebar.file_uploader("Credenziali Google (JSON)", type=["json"], disabled=not use_gsheet)

# Verifica API Key
if api_key:
    openai.api_key = api_key
    st.sidebar.success("✅ API Key caricata")
else:
    st.sidebar.warning("⚠️ Inserisci la tua OpenAI API Key per continuare")

# Caricamento file CSV
uploaded_file = st.file_uploader("📤 Carica il file CSV con i prodotti")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Anteprima:")
    st.dataframe(df.head())

    # Calcola token stimati
    def estimate_token_cost(data):
        total_tokens = 0
        for _, row in data.iterrows():
            content = " ".join([str(v) for k, v in row.items() if k.lower() not in ["description", "short_description"]])
            total_tokens += len(content.split()) * 1.5  # stima
        return int(total_tokens), round((total_tokens / 1000) * COST_PER_1K_TOKENS, 4)

    tokens, cost = estimate_token_cost(df)
    st.info(f"📈 Token stimati: **{tokens}** | Costo stimato: **${cost}**")

    confirm = st.button("✅ Conferma e Genera Descrizioni")

    # Connessione Google Sheets
    def connect_to_gsheet(json_file, sheet_id):
        credentials = Credentials.from_service_account_info(json.load(json_file),
                                                             scopes=["https://www.googleapis.com/auth/spreadsheets"])
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet

    def save_to_gsheet(dataframe, sheet):
        sheet.clear()
        sheet.update([dataframe.columns.values.tolist()] + dataframe.values.tolist())

    # Funzione generazione descrizioni
    def generate_descriptions(row):
        if not api_key:
            return "API key mancante", "API key mancante", 0

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

        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            result = response.choices[0].message.content.strip()
            parts = [p.strip("1234567890.-: \n") for p in result.split("\n") if p.strip()]
            long_desc = parts[0] if len(parts) > 0 else "Descrizione non trovata"
            short_desc = parts[1] if len(parts) > 1 else long_desc[:100]
            used_tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
            return long_desc, short_desc, used_tokens
        except Exception as e:
            return f"Errore: {e}", "Errore", 0

    # Avvia generazione
    if confirm:
        if not api_key:
            st.error("❌ Nessuna API Key inserita.")
        else:
            st.success("🚀 Generazione in corso...")
            results = []
            progress = st.progress(0)
            total = len(df)

            for idx, row in df.iterrows():
                desc, short, tokens_used = generate_descriptions(row)
                row["description"] = desc
                row["short_description"] = short
                row["token_used"] = tokens_used
                results.append(row)
                progress.progress((idx + 1) / total)
                time.sleep(0.5)

            result_df = pd.DataFrame(results)
            st.success("✅ Descrizioni generate con successo!")
            st.dataframe(result_df[["description", "short_description"]].head())

            # Salvataggio Google Sheet
            if use_gsheet and json_file and sheet_id:
                try:
                    sheet = connect_to_gsheet(json_file, sheet_id)
                    save_to_gsheet(result_df, sheet)
                    st.success("📌 Storico salvato su Google Sheets")
                except Exception as e:
                    st.error(f"Errore salvataggio su Google Sheets: {e}")

            # Download file
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Scarica CSV aggiornato", csv_out, file_name="prodotti_con_descrizioni.csv")
