import streamlit as st
import pandas as pd
from io import StringIO
from openai import OpenAI
import os

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Generatore Descrizioni", layout="centered")

# --- INSERISCI LA TUA CHIAVE API QUI (o lasciala vuota per inserimento manuale) ---
API_KEY = ""  # es: "sk-..."

st.title("📝 Generatore Descrizioni Prodotto (OpenAI GPT-3.5)")
st.write("Carica un file CSV contenente i tuoi prodotti. Verranno generate le colonne `description` e `short_description`.")

# Inserimento API se non è nel codice
if not API_KEY:
    API_KEY = st.text_input("Inserisci la tua API Key OpenAI", type="password")

if API_KEY:
    st.success("✅ API Key inserita correttamente")

# --- Caricamento File ---
uploaded_file = st.file_uploader("Carica il tuo file CSV", type=["csv"])

def generate_descriptions(row, api_key):
    client = OpenAI(api_key=api_key)

    product_info = ", ".join([
        f"{k}: {v}" for k, v in row.items()
        if k.lower() not in ["description", "short_description"] and pd.notna(v)
    ])

    prompt = f"""
Scrivi due descrizioni per una calzatura da vendere online:
1. Una descrizione lunga di circa 60 parole.
2. Una descrizione breve di circa 20 parole.
Usa un tono accattivante, caldo, professionale, user friendly e SEO friendly.
Dettagli del prodotto: {product_info}
    """

    try:
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
    except Exception as e:
        return f"Errore: {e}", f"Errore: {e}"

# --- Elaborazione file ---
if uploaded_file and API_KEY:
    df = pd.read_csv(uploaded_file)

    if "description" not in df.columns:
        df["description"] = ""
    if "short_description" not in df.columns:
        df["short_description"] = ""

    st.info(f"🔍 File caricato correttamente con {len(df)} righe.")

    if st.button("🚀 Genera Descrizioni"):
        progress = st.progress(0)
        for idx, row in df.iterrows():
            long_desc, short_desc = generate_descriptions(row, API_KEY)
            df.at[idx, "description"] = long_desc
            df.at[idx, "short_description"] = short_desc
            progress.progress((idx + 1) / len(df))
        st.success("✅ Descrizioni generate con successo!")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Scarica il file con le descrizioni",
            data=csv,
            file_name="prodotti_con_descrizioni.csv",
            mime="text/csv",
        )
