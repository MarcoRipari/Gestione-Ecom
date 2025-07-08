
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import random

st.set_page_config(page_title="Descrizioni Calzature - Transformers", layout="centered")
st.title("🤖 Generatore Descrizioni con AI (BERT semantico)")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

KEY_TERMS = ["model", "title", "material", "color", "style", "name", "type", "gender", "category"]

def colonne_rilevanti(colonne):
    chiavi = model.encode(KEY_TERMS, convert_to_tensor=True)
    col_embed = model.encode(colonne, convert_to_tensor=True)
    sim = util.cos_sim(col_embed, chiavi).max(dim=1).values
    return [col for col, s in zip(colonne, sim) if s > 0.5]

def pulisci_valore(val):
    return str(val).replace("_", " ").replace("-", " ").strip().capitalize()

def genera_descrizioni(row, colonne_rilevanti):
    intro = random.choice([
        "Pensata per chi cerca stile e funzionalità",
        "Un equilibrio perfetto tra design e comfort",
        "Una calzatura per distinguersi ogni giorno",
    ])
    finale = random.choice([
        "Perfetta per qualsiasi occasione.",
        "Un must-have nel guardaroba di stagione.",
        "Ideale per outfit dinamici e casual.",
    ])

    elementi = []
    for col in colonne_rilevanti:
        val = row[col]
        if pd.notna(val):
            elementi.append(pulisci_valore(val))

    descrizione_lunga = f"{intro}. " + ", ".join(elementi[:8]) + f". {finale}"
    descrizione_breve = ", ".join(elementi[:4]) + "."

    return pd.Series([descrizione_lunga[:450], descrizione_breve[:150]])

uploaded_file = st.file_uploader("📤 Carica il tuo file CSV", type=["csv"])

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
        st.download_button("📥 Scarica CSV con descrizioni", data=csv, file_name="output_transformers.csv", mime="text/csv")
