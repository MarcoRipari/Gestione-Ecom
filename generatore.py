import streamlit as st
import pandas as pd
import openai
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === CONFIGURAZIONE ===
st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="wide")
openai.api_key = "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"

# === CONNESSIONE GOOGLE SHEETS ===
def connect_to_gsheet(json_keyfile, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile, scope)
    client = gspread.authorize(creds)
    return client.open(sheet_name).sheet1

# === PROMPT BASE ===
def generate_description(product_info):
    prompt = f"""
Scrivi due descrizioni per una calzatura da vendere online, usando un tono accattivante, caldo, professionale, SEO e user friendly.
- Una descrizione lunga di circa 60 parole
- Una descrizione breve di circa 20 parole

Informazioni prodotto:
{product_info}

Rispondi nel formato:
LONG: ...
SHORT: ...
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip()
    long_desc, short_desc = "", ""
    for line in text.splitlines():
        if line.lower().startswith("long:"):
            long_desc = line[5:].strip()
        elif line.lower().startswith("short:"):
            short_desc = line[6:].strip()
    return long_desc, short_desc

# === INTERFACCIA ===
st.title("🥿 Generatore di Descrizioni Calzature + Storico su Google Sheets")

# Carica CSV
uploaded_file = st.file_uploader("📁 Carica il file CSV dei prodotti", type=["csv"])

# Credenziali
json_keyfile = st.file_uploader("🔐 Carica il file JSON di credenziali Google", type=["json"])
sheet_name = st.text_input("📄 Nome del Google Sheet dove salvare lo storico", value="storico_descrizioni")

if uploaded_file and json_keyfile and sheet_name:
    df = pd.read_csv(uploaded_file)
    gsheet = connect_to_gsheet(json_keyfile.name, sheet_name)

    if "description" not in df.columns:
        df["description"] = ""
    if "short_description" not in df.columns:
        df["short_description"] = ""

    st.success("✅ File e credenziali caricati correttamente. Pronto a generare.")
    if st.button("🚀 Genera descrizioni e salva nello storico"):
        for i, row in df.iterrows():
            if not row["description"] or not row["short_description"]:
                product_data = ", ".join([f"{col}: {row[col]}" for col in df.columns if col not in ["description", "short_description"]])
                long_desc, short_desc = generate_description(product_data)
                df.at[i, "description"] = long_desc
                df.at[i, "short_description"] = short_desc
                gsheet.append_row([row.get(col, "") for col in df.columns] + [long_desc, short_desc])
        st.success("✅ Descrizioni generate e salvate su Google Sheets!")
        st.dataframe(df)

        # Download del nuovo CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica CSV aggiornato", csv, "prodotti_con_descrizioni.csv", "text/csv")

