import streamlit as st
import pandas as pd
import openai
import json
import gspread
from io import StringIO

# Imposta qui la tua API Key
openai.api_key = "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"

st.set_page_config(page_title="Generatore Descrizioni", layout="wide")
st.title("🧠 Generatore Descrizioni Calzature con OpenAI + Google Sheets")

# Caricamento credenziali Google
credentials_file = st.file_uploader("📄 Carica il file `credentials.json`", type="json")
sheet_id = st.text_input("🔗 Inserisci lo Sheet ID (da URL Google Sheets)")

# Caricamento CSV
csv_file = st.file_uploader("📦 Carica il file CSV dei prodotti", type="csv")

# Stima token semplice
def count_tokens_approx(text):
    return len(text) // 4  # stima empirica (1 token ≈ 4 caratteri)

# Generazione descrizioni
def generate_descriptions(row):
    base_info = " ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
    prompt = f"""Scrivi due descrizioni per un prodotto di calzature:
1. Una descrizione lunga di circa 60 parole.
2. Una descrizione breve di circa 20 parole.
Tono accattivante, caldo, professionale, user-friendly e SEO-friendly. Dettagli prodotto: {base_info}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )

    output = response.choices[0].message.content.strip()
    parts = output.split("\n")
    long_desc = ""
    short_desc = ""

    for p in parts:
        if "1." in p or "Descrizione lunga" in p:
            long_desc = p.split(":", 1)[-1].strip()
        elif "2." in p or "Descrizione breve" in p:
            short_desc = p.split(":", 1)[-1].strip()

    return long_desc, short_desc, count_tokens_approx(prompt + output)

# Connessione a Google Sheets
def connect_to_gsheet(json_file, sheet_id):
    credentials = json.load(json_file)
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_key(sheet_id)
    return sh.sheet1

# Corpo principale
if credentials_file and sheet_id and csv_file:
    try:
        df = pd.read_csv(csv_file)
        st.success("✅ File CSV caricato correttamente!")

        st.subheader("🔍 Anteprima dati")
        st.dataframe(df.head())

        if st.button("💰 Stima costo generazione"):
            total_tokens = 0
            for _, row in df.iterrows():
                base_info = " ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                prompt = f"""Scrivi due descrizioni per un prodotto di calzature:
1. Una descrizione lunga di circa 60 parole.
2. Una descrizione breve di circa 20 parole.
Tono accattivante, caldo, professionale, user-friendly e SEO-friendly. Dettagli prodotto: {base_info}"""
                total_tokens += count_tokens_approx(prompt)

            cost = (total_tokens / 1000) * 0.001  # GPT-3.5 input token cost
            st.info(f"🔢 Token stimati: {total_tokens}")
            st.warning(f"💲 Costo stimato: circa ${cost:.4f}")

            if st.button("🚀 Conferma e Genera descrizioni"):
                long_descs = []
                short_descs = []
                token_used = []

                with st.spinner("Generazione in corso..."):
                    for idx, row in df.iterrows():
                        long, short, tokens = generate_descriptions(row)
                        long_descs.append(long)
                        short_descs.append(short)
                        token_used.append(tokens)

                df["description"] = long_descs
                df["short_description"] = short_descs
                df["tokens"] = token_used

                st.success("✅ Descrizioni generate con successo!")
                st.dataframe(df[["description", "short_description"]].head())

                if st.button("📤 Salva su Google Sheets"):
                    sheet = connect_to_gsheet(credentials_file, sheet_id)
                    sheet.clear()
                    sheet.update([df.columns.values.tolist()] + df.values.tolist())
                    st.success("✅ Dati salvati su Google Sheets!")

    except Exception as e:
        st.error(f"❌ Errore: {e}")
