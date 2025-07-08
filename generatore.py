import streamlit as st
import pandas as pd
import openai
import io

st.set_page_config(page_title="Generatore Descrizioni AI", layout="wide")
st.title("📝 Generatore Descrizioni Prodotto con OpenAI")

# Inserimento chiave API OpenAI
api_key = st.text_input("🔐 Inserisci la tua OpenAI API Key", type="password")
uploaded_file = st.file_uploader("📤 Carica un file CSV con i prodotti", type=["csv"])

# Funzione per chiamare OpenAI GPT
def call_openai_gpt(prompt, api_key):
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un copywriter esperto in e-commerce. Genera descrizioni accattivanti, SEO-friendly, varie e professionali per le schede prodotto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore OpenAI: {e}"

# Quando file e chiave sono caricati
if uploaded_file and api_key:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Errore nella lettura del file: {e}")
        st.stop()

    st.success("✅ File caricato correttamente!")
    st.write("Anteprima del file:", df.head())

    # Nomi colonne da creare o aggiornare
    desc_col = "description"
    short_col = "short_description"

    if desc_col not in df.columns:
        df[desc_col] = ""
    if short_col not in df.columns:
        df[short_col] = ""

    total = len(df)
    if total == 0:
        st.warning("⚠️ Il file non contiene righe.")
        st.stop()
    progress = st.progress(0)

    for idx, row in df.iterrows():
        try:
            # Costruzione prompt con tutte le info disponibili
            product_info = ", ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notnull(row[col]) and str(row[col]).strip() != ""]
            )
            prompt = (
                f"Genera una descrizione lunga di circa 60 parole (+/-10%) per il seguente prodotto:\n{product_info}\n\n"
                "Poi genera una descrizione breve di circa 20 parole (+/-10%). "
                "Usa uno stile accattivante, SEO-friendly, ma varia i termini tra i prodotti. "
                "Restituisci prima la descrizione lunga, poi quella breve, separate da |||"
            )

            result = call_openai_gpt(prompt, api_key)
            if "|||" in result:
                long_desc, short_desc = result.split("|||", 1)
            else:
                long_desc, short_desc = result, ""

            df.at[idx, desc_col] = long_desc.strip()
            df.at[idx, short_col] = short_desc.strip()

        except Exception as e:
            st.error(f"❌ Errore alla riga {idx}: {e}")
        progress.progress((idx + 1) / total)

    st.success("✅ Descrizioni generate con successo!")

    # Esportazione del file aggiornato
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Scarica il CSV con le descrizioni",
        data=csv_buffer.getvalue(),
        file_name="prodotti_con_descrizioni.csv",
        mime="text/csv"
    )
