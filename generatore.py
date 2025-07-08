import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Generatore Descrizioni AI - Mistral", layout="centered")
st.title("🧠 Generatore Descrizioni Prodotto")
st.markdown("Usa **Mistral-7B** via OpenRouter per creare descrizioni SEO-friendly e accattivanti.")

# Inserimento chiave API OpenRouter
api_key = st.text_input("🔑 Inserisci la tua API Key OpenRouter", type="password")

uploaded_file = st.file_uploader("📤 Carica il file CSV con i prodotti", type="csv")

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)
    df.fillna("", inplace=True)

    # Colonne output
    df["description"] = ""
    df["short_description"] = ""

    st.info("Generazione in corso. Potrebbero volerci alcuni secondi...")
    progress = st.progress(0)
    total = len(df)

    for idx, row in df.iterrows():
        # Crea prompt dinamico per descrizione
        row_data = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if str(row[col]).strip()])
        prompt = f"""
Agisci come un copywriter professionista di un e-commerce di calzature.
Genera due descrizioni per una scheda prodotto basandoti sulle informazioni seguenti:
{row_data}

1. Descrizione lunga (60 parole circa, tono caldo, professionale, SEO e user-friendly):
2. Descrizione breve (20 parole circa, diretta e accattivante):
""".strip()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mistral/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "Sei un esperto di copywriting e scrittura SEO per schede prodotto."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))
            result = response.json()
            output = result["choices"][0]["message"]["content"]

            # Parsing output
            parts = output.split("2. ")
            desc_lunga = parts[0].replace("1. ", "").strip()
            desc_breve = parts[1].strip() if len(parts) > 1 else ""

            df.at[idx, "description"] = desc_lunga
            df.at[idx, "short_description"] = desc_breve

        except Exception as e:
            st.error(f"Errore alla riga {idx}: {e}")
            continue

        progress.progress((idx + 1) / total)

    st.success("✅ Descrizioni generate con successo!")
    st.download_button("📥 Scarica CSV con descrizioni", data=df.to_csv(index=False), file_name="prodotti_con_descrizioni.csv", mime="text/csv")
