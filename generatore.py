import streamlit as st
import pandas as pd
import openai
import io

st.set_page_config(page_title="Generatore Descrizioni AI", layout="wide")
st.title("📝 Generatore Descrizioni Prodotto con OpenAI")

# API Key
if "api_key" not in st.session_state:
    st.session_state.api_key = st.text_input("🔐 Inserisci la tua OpenAI API Key", type="password")
else:
    st.text_input("🔐 OpenAI API Key (già impostata)", value=st.session_state.api_key, type="password")

uploaded_file = st.file_uploader("📤 Carica un file CSV", type="csv")

# Toni dinamici
category_tone_map = {
    "sneakers": "Tono sportivo e moderno. Enfatizza comfort e stile.",
    "stivali": "Tono deciso. Sottolinea robustezza e protezione.",
    "sandali": "Tono fresco ed estivo. Valorizza leggerezza e libertà.",
    "bambino": "Tono giocoso e sicuro. Parla di comfort e protezione.",
    "eleganti": "Tono raffinato. Metti in risalto stile e classe."
}

# Funzione chiamata OpenAI
def call_openai(prompt):
    try:
        openai.api_key = st.session_state.api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sei un copywriter esperto di e-commerce. Scrivi descrizioni efficaci, SEO friendly e uniche."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore OpenAI: {e}"

# Stima semplice dei token e costo
def estimate_tokens_and_cost(df):
    total_chars = 0
    for _, row in df.iterrows():
        product_data = ", ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
        total_chars += len(product_data)
    total_tokens = int(total_chars / 4) + len(df) * 80  # output stimato 80 token/descrizione
    cost = (total_tokens * 0.001)  # prompt+completion semplificato
    return total_tokens, cost

if uploaded_file and st.session_state.api_key:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        st.stop()

    st.success("✅ File caricato correttamente!")
    st.write(df.head())

    if "description" not in df.columns:
        df["description"] = ""
    if "short_description" not in df.columns:
        df["short_description"] = ""

    total_tokens, estimated_cost = estimate_tokens_and_cost(df)
    st.info(f"🔢 Token stimati: {total_tokens:,} — 💰 Costo stimato: ~${estimated_cost:.4f} (GPT-3.5)")

    if st.button("✅ Genera descrizioni con OpenAI"):
        progress = st.progress(0)
        total = len(df)

        for i, row in df.iterrows():
            try:
                product_data = ", ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
                category_text = " ".join([str(row[col]).lower() for col in df.columns if "categoria" in col.lower()])
                tone = ""
                for key in category_tone_map:
                    if key in category_text:
                        tone = category_tone_map[key]
                        break

                prompt = (
                    f"Genera una descrizione lunga (circa 60 parole) e una breve (circa 20 parole) per questo prodotto:\n"
                    f"{product_data}\n\n"
                    f"{tone} Rispondi nel formato: DESCRIZIONE LUNGA ||| DESCRIZIONE BREVE"
                )

                output = call_openai(prompt)
                if "|||" in output:
                    long_desc, short_desc = output.split("|||")
                    df.at[i, "description"] = long_desc.strip()
                    df.at[i, "short_description"] = short_desc.strip()
                else:
                    df.at[i, "description"] = output.strip()
                    df.at[i, "short_description"] = ""

            except Exception as e:
                st.error(f"Errore alla riga {i}: {e}")
            progress.progress((i + 1) / total)

        st.success("✅ Descrizioni generate!")
        csv = io.StringIO()
        df.to_csv(csv, index=False)
        st.download_button("📥 Scarica CSV aggiornato", csv.getvalue(), "output.csv", "text/csv")
