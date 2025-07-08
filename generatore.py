import streamlit as st
import pandas as pd
import re
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import tempfile
import json

# === CONFIG ===
OPENAI_API_KEY = "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"
SHEET_ID = "1R9diynXeS4zTzKnjz1Kp1Uk7MNJX8mlHgV82NBvjXZc"  # il tuo ID Google Sheet

CREDENTIALS_JSON = {
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

# === SETUP ===
st.set_page_config(page_title="Generatore Descrizioni", layout="centered")
st.title("📝 Generatore Descrizioni Prodotto")

# === GOOGLE SHEET ===
def connect_to_gsheet(credentials_dict, sheet_id):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as temp:
        json.dump(credentials_dict, temp)
        temp.flush()
        creds = ServiceAccountCredentials.from_json_keyfile_name(temp.name, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    return sheet

# === FUNZIONE OPENAI ===
def generate_descriptions(row):
    client = OpenAI(api_key=OPENAI_API_KEY)

    product_info = ", ".join([
        f"{k}: {v}" for k, v in row.items()
        if k.lower() not in ["description", "short_description"] and pd.notna(v)
    ])

    prompt = f"""
Scrivi DUE descrizioni per una calzatura da vendere online.

Formato:
###DESCRIZIONE_LUNGA###
[Inserisci qui una descrizione lunga di circa 60 parole]
###DESCRIZIONE_CORTA###
[Inserisci qui una descrizione breve di circa 20 parole]

Tono: accattivante, professionale, user friendly, SEO friendly.

Dati prodotto: {product_info}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()

        # Parsing basato su marcatori ###
        long_desc = short_desc = "Descrizione non trovata"
        if "###DESCRIZIONE_LUNGA###" in content and "###DESCRIZIONE_CORTA###" in content:
            parts = content.split("###DESCRIZIONE_CORTA###")
            long_part = parts[0].replace("###DESCRIZIONE_LUNGA###", "").strip()
            short_part = parts[1].strip()
            long_desc = long_part if long_part else "Descrizione lunga non trovata"
            short_desc = short_part if short_part else "Descrizione corta non trovata"

        return long_desc, short_desc

    except Exception as e:
        return f"Errore: {e}", f"Errore: {e}"

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📤 Carica un file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "SKU" not in df.columns:
        st.error("❌ Il file deve contenere una colonna chiamata 'SKU'")
    else:
        st.dataframe(df.head())

        st.markdown("---")

        if st.button("📊 Stima costo descrizioni"):
            num_prodotti = len(df)
            tokens_per_desc = 150  # media stimata per 2 descrizioni
            total_tokens = num_prodotti * tokens_per_desc
            cost = (total_tokens / 1000) * 0.0015  # prezzo stimato gpt-3.5

            st.info(f"🧮 Totale prodotti: {num_prodotti}")
            st.info(f"💰 Costo stimato: ~{cost:.4f} USD")

        if st.button("🚀 Conferma e genera"):
            st.info("🔄 Generazione in corso...")
            progress = st.progress(0)

            generated = []
            for idx, row in df.iterrows():
                long_desc, short_desc = generate_descriptions(row)
                generated.append({
                    "SKU": row["SKU"],
                    "Descrizione": long_desc,
                    "Descrizione Corta": short_desc
                })
                progress.progress((idx + 1) / len(df))

            result_df = pd.DataFrame(generated)

            # SALVA SU GOOGLE SHEETS
            try:
                sheet = connect_to_gsheet(CREDENTIALS_JSON, SHEET_ID)
                worksheet = sheet.sheet1

                existing_data = worksheet.get_all_values()
                if not existing_data:
                    # Foglio vuoto → intestazione + righe
                    worksheet.append_rows([result_df.columns.values.tolist()] + result_df.values.tolist())
                else:
                    # Foglio ha già dati → aggiungi solo nuove righe
                    worksheet.append_rows(result_df.values.tolist())

                st.success("✅ Descrizioni generate e aggiunte a Google Sheets!")
            except Exception as e:
                st.error(f"Errore salvataggio su Google Sheets: {e}")

            # CSV DOWNLOAD
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Scarica il CSV",
                data=csv,
                file_name="descrizioni_generate.csv",
                mime="text/csv"
            )
