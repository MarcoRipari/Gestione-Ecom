import streamlit as st
import pandas as pd
import openai

# Imposta la tua chiave OpenAI direttamente nel codice
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

st.title("📝 Generatore Descrizioni Prodotto (Calzature)")

# Messaggio di conferma che l'API è attiva
if OPENAI_API_KEY.startswith("sk-"):
    st.success("✅ OpenAI API key impostata correttamente.")

# Carica il file CSV
uploaded_file = st.file_uploader("📁 Carica il file CSV dei prodotti", type=["csv"])

# Funzione per contare i token stimati
def count_tokens(text: str) -> int:
    # Stima approssimativa: 1 token ≈ 4 caratteri
    return int(len(text) / 4)

# Funzione per generare la descrizione
def generate_description(row):
    prompt = f"""
Scrivi una descrizione lunga (60 parole) e una descrizione breve (20 parole) per una calzatura da vendere online.
Usa un tono accattivante, caldo, professionale e SEO-friendly. Alterna lo stile tra i vari prodotti per evitare ripetizioni.

Dati Prodotto:
- Nome: {row.get("nome", "")}
- Brand: {row.get("brand", "")}
- Colore: {row.get("colore", "")}
- Genere: {row.get("genere", "")}
- Categoria: {row.get("categoria", "")}
- Materiale: {row.get("materiale", "")}
- Prezzo: {row.get("prezzo", "")}

Restituisci solo:
DESCRIZIONE: ...
SHORT: ...
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    output = response.choices[0].message.content.strip()
    descrizione = ""
    breve = ""

    for line in output.split("\n"):
        if line.upper().startswith("DESCRIZIONE"):
            descrizione = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SHORT"):
            breve = line.split(":", 1)[1].strip()

    return descrizione, breve, count_tokens(prompt)

# Logica principale
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "description" not in df.columns:
        df["description"] = ""
    if "short_description" not in df.columns:
        df["short_description"] = ""

    total_tokens = 0
    st.info("🚀 Generazione in corso, attendi qualche secondo...")
    progress = st.progress(0)

    for idx, row in df.iterrows():
        descrizione, breve, tokens = generate_description(row)
        df.at[idx, "description"] = descrizione
        df.at[idx, "short_description"] = breve
        total_tokens += tokens
        progress.progress((idx + 1) / len(df))

    costo_usd = (total_tokens / 1000) * 0.001
    st.success(f"✅ Completato! Token totali: {total_tokens} | Costo stimato: ${costo_usd:.4f}")

    st.download_button(
        label="💾 Scarica il file con descrizioni",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="prodotti_descrizioni.csv",
        mime="text/csv"
    )
