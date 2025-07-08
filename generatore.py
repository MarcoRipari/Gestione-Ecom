
import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="Generatore Descrizioni AI", layout="wide")

st.title("📝 Generatore Descrizioni Prodotto con Mixtral AI")

api_key = st.text_input("🔐 Inserisci la tua API Key di OpenRouter", type="password")

uploaded_file = st.file_uploader("📤 Carica un file CSV", type=["csv"])

def call_openrouter_mixtral(prompt, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "Sei un copywriter esperto in e-commerce. Genera descrizioni accattivanti, SEO-friendly, diverse e professionali."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.9
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Errore API: {response.status_code} - {response.text}"

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)

    st.success("✅ File caricato correttamente.")
    st.write("Anteprima del file:", df.head())

    desc_col = "description"
    short_col = "short_description"

    if desc_col not in df.columns:
        df[desc_col] = ""
    if short_col not in df.columns:
        df[short_col] = ""

    progress = st.progress(0)
    total = len(df)

    for idx, row in df.iterrows():
        product_info = ", ".join([f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col])])
        prompt = (
            f"Genera una descrizione lunga di 60 parole (+/-10%) per il seguente prodotto:
{product_info}

"
            "Poi genera anche una descrizione breve di circa 20 parole (+/-10%). "
            "Lo stile deve essere accattivante, SEO-friendly, coerente ma vario tra i prodotti. "
            "Restituisci prima la descrizione lunga, poi quella breve separate da |||"
        )
        result = call_openrouter_mixtral(prompt, api_key)
        if "|||" in result:
            long_desc, short_desc = result.split("|||", 1)
        else:
            long_desc, short_desc = result, ""

        df.at[idx, desc_col] = long_desc.strip()
        df.at[idx, short_col] = short_desc.strip()
        progress.progress((idx + 1) / total)

    st.success("✅ Descrizioni generate con successo!")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Scarica il file con descrizioni", csv_buffer.getvalue(), file_name="prodotti_con_descrizioni.csv", mime="text/csv")
