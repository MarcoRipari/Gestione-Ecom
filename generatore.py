
import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="centered")
st.title("🤖 Generatore Intelligente Descrizioni Calzature")

st.markdown("Carica un CSV: il sistema analizzerà automaticamente i dati, individuerà le informazioni più rilevanti e genererà descrizioni lunghe e brevi, ottimizzate per SEO.")

# Frasi dinamiche per varietà testuale
INTROS = [
    "Scopri un design essenziale ma ricco di personalità",
    "Una calzatura che fonde stile e comfort",
    "Eleganza e praticità per accompagnarti ogni giorno",
    "Un modello pensato per chi ama distinguersi",
    "Stile versatile che valorizza ogni look"
]

FINALI = [
    "Perfetta per outfit casual e smart.",
    "Ideale per tutte le stagioni.",
    "Un must-have per il guardaroba contemporaneo.",
    "Pensata per chi non rinuncia al comfort con stile.",
    "Accompagna ogni passo con personalità."
]

def pulisci_valore(val):
    return str(val).replace("_", " ").replace("-", " ").strip().capitalize()

def genera_descrizioni_intelligenti(row, colonne_rilevanti):
    # Priorità: nome > categoria > materiale > colore > altri
    pesi = {
        "name": 5, "title": 5, "modello": 4, "categoria": 3, 
        "category": 3, "materiale": 3, "material": 3,
        "colore": 2, "color": 2
    }

    elementi = []

    for col in colonne_rilevanti:
        val = row[col]
        if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 1:
            peso = max([pesi[k] for k in pesi if k in col.lower()] + [1])
            elementi.extend([pulisci_valore(val)] * peso)

    intro = random.choice(INTROS)
    finale = random.choice(FINALI)

    descrizione_lunga = f"{intro}. " + ", ".join(elementi[:10]) + f". {finale}"
    descrizione_breve = ", ".join(elementi[:4]) + "."

    if len(descrizione_lunga.split()) > 66:
        descrizione_lunga = " ".join(descrizione_lunga.split()[:66]) + "..."
    if len(descrizione_breve.split()) > 22:
        descrizione_breve = " ".join(descrizione_breve.split()[:22]) + "..."

    return pd.Series([descrizione_lunga, descrizione_breve])

uploaded_file = st.file_uploader("📤 Carica un file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')

    # Seleziona solo colonne testuali e rilevanti
    colonne_testuali = [
        col for col in df.columns
        if df[col].dtype == object and not col.lower().startswith(("id", "sku", "ean"))
    ]

    if not colonne_testuali:
        st.error("❌ Nessuna colonna testuale trovata nel file.")
    else:
        st.markdown(f"**Colonne utilizzate per la generazione:** {', '.join(colonne_testuali)}")

        df[["description", "short_description"]] = df.apply(
            lambda row: genera_descrizioni_intelligenti(row, colonne_testuali), axis=1
        )

        st.success("✅ Descrizioni generate con successo!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
        st.download_button(
            label="📥 Scarica il CSV aggiornato",
            data=csv,
            file_name='prodotti_descritti.csv',
            mime='text/csv'
        )
