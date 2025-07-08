
import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="centered")
st.title("📝 Generatore di Descrizioni per Calzature")

st.markdown("Carica un file CSV contenente informazioni sui prodotti. Le descrizioni verranno generate usando **tutte le colonne** disponibili nella riga.")

# Blocchi di testo ispirati allo stile originale
intro_frasi = [
    "Un design contemporaneo pensato per distinguersi",
    "Linee essenziali e stile intramontabile",
    "Versatilità e comfort in un solo modello",
    "Eleganza urbana con dettagli curati",
    "Perfetta per ogni occasione quotidiana",
    "Un modello che coniuga stile e funzionalità"
]

finale_frasi = [
    "Si abbina perfettamente a look casual o più ricercati.",
    "Perfetta per completare ogni outfit con personalità.",
    "Un tocco di stile da mattina a sera.",
    "Pensata per uno stile pratico e raffinato.",
    "La scelta giusta per un look urbano e dinamico."
]

def generate_descriptions_from_row(row):
    elementi = []
    for col, val in row.items():
        if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 1:
            elementi.append(str(val).strip())

    intro = random.choice(intro_frasi)
    finale = random.choice(finale_frasi)

    descrizione_lunga = f"{intro}. " + ", ".join(elementi[:6]) + f". {finale}"
    descrizione_breve = ", ".join(elementi[:3]) + "."

    # Trim per rispettare le lunghezze approssimative
    if len(descrizione_lunga.split()) > 66:
        descrizione_lunga = " ".join(descrizione_lunga.split()[:66]) + "..."
    if len(descrizione_breve.split()) > 22:
        descrizione_breve = " ".join(descrizione_breve.split()[:22]) + "..."

    return pd.Series([descrizione_lunga, descrizione_breve])

uploaded_file = st.file_uploader("Carica il file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    st.markdown(f"**Colonne trovate:** {', '.join(df.columns)}")

    output_long = st.text_input("Nome colonna per la descrizione lunga", value="description")
    output_short = st.text_input("Nome colonna per la descrizione breve", value="short_description")

    if st.button("Genera Descrizioni"):
        df[[output_long, output_short]] = df.apply(generate_descriptions_from_row, axis=1)
        st.success("Descrizioni generate con successo!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
        st.download_button(
            label="📥 Scarica il CSV con le descrizioni",
            data=csv,
            file_name='descrizioni_prodotti.csv',
            mime='text/csv'
        )
