import streamlit as st
import pandas as pd
import spacy
import random

st.set_page_config(page_title="Generatore Descrizioni - spaCy", layout="centered")
st.title("👟 Generatore Descrizioni Calzature (AI leggera - spaCy)")

st.markdown("Carica un file CSV: useremo NLP per selezionare solo le colonne utili e generare descrizioni complete e SEO-friendly.")

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()

KEY_TERMS = ["model", "name", "title", "style", "material", "color", "shape", "category", "gender", "type"]

def colonne_rilevanti(colonne):
    soglia = 0.65
    col_rilevanti = []
    for col in colonne:
        col_doc = nlp(col.lower())
        max_sim = max(nlp(term).similarity(col_doc) for term in KEY_TERMS)
        if max_sim > soglia:
            col_rilevanti.append(col)
    return col_rilevanti

def pulisci_valore(val):
    return str(val).replace("_", " ").replace("-", " ").strip().capitalize()

def genera_descrizioni(row, colonne_rilevanti):
    intro = random.choice([
        "Stile, praticità e comfort",
        "Un mix perfetto tra eleganza e funzionalità",
        "Un modello pensato per ogni occasione",
        "Perfette per accompagnarti tutto il giorno"
    ])
    finale = random.choice([
        "Ideali per un look dinamico e moderno.",
        "Perfette per look casual e raffinati.",
        "Immancabili nel guardaroba di stagione."
    ])

    elementi = []
    for col in colonne_rilevanti:
        val = row[col]
        if pd.notna(val):
            elementi.append(pulisci_valore(val))

    descrizione_lunga = f"{intro}. " + ", ".join(elementi[:8]) + f". {finale}"
    descrizione_breve = ", ".join(elementi[:4]) + "."

    return pd.Series([descrizione_lunga[:450], descrizione_breve[:150]])

uploaded_file = st.file_uploader("📤 Carica un file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    colonne_testuali = [col for col in df.columns if df[col].dtype == object]
    col_ril = colonne_rilevanti(colonne_testuali)

    if not col_ril:
        st.warning("Nessuna colonna rilevante trovata.")
    else:
        st.markdown(f"✅ Colonne utilizzate: {', '.join(col_ril)}")
        df[["description", "short_description"]] = df.apply(lambda r: genera_descrizioni(r, col_ril), axis=1)
        st.dataframe(df.head())

        csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
        st.download_button("📥 Scarica CSV aggiornato", data=csv, file_name="output_spacy.csv", mime="text/csv")
