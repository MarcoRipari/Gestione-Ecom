
import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="Generatore Descrizioni Calzature", layout="centered")
st.title("📝 Generatore di Descrizioni per Calzature")

st.markdown("Carica un file CSV con i campi come `name`, `material`, `category`. Potrai anche scegliere dove inserire le descrizioni generate.")

# Blocchi coerenti con lo stile originale
introduzioni = [
    "Perfetta per completare ogni outfit",
    "Pensata per uno stile moderno e dinamico",
    "Design essenziale e sempre attuale",
    "Look contemporaneo con dettagli ricercati",
    "Ideale per un uso quotidiano con stile",
    "Versatilità e carattere in un solo modello"
]

dettagli_tomaia = [
    "tomaia in {materiale} con {dettaglio}",
    "realizzata in {materiale}, impreziosita da {dettaglio}",
    "struttura in {materiale} con finiture {finitura}",
    "tomaia {materiale} rifinita con {dettaglio}"
]

dettagli_suola = [
    "suola in {suola} per massimo comfort",
    "base in {suola} che assicura aderenza e stabilità",
    "dotata di suola in {suola}, ideale per l'uso urbano",
    "supporto in {suola} pensato per la quotidianità"
]

conclusioni = [
    "Perfetta con jeans, pantaloni slim o gonne midi.",
    "Si abbina facilmente ad ogni look del giorno.",
    "Un tocco di stile da mattina a sera.",
    "Un alleato di stile in ogni stagione."
]

usi = ['urbano', 'casual', 'giornaliero', 'versatile']
dettagli = ['cuciture in rilievo', 'inserti a contrasto', 'linee pulite', 'dettagli tono su tono', 'cuciture visibili']
finiture = ['lucide', 'opache', 'minimal', 'moderne']
suole = ['gomma', 'poliuretano', 'TR']

def generate_descriptions(row, name_col, material_col, category_col):
    nome = row.get(name_col, 'Modello')
    materiale = row.get(material_col, 'materiale tecnico')
    categoria = row.get(category_col, '').capitalize()

    intro = random.choice(introduzioni)
    tomaia = random.choice(dettagli_tomaia).format(materiale=materiale, dettaglio=random.choice(dettagli), finitura=random.choice(finiture))
    suola = random.choice(dettagli_suola).format(suola=random.choice(suole))
    finale = random.choice(conclusioni)

    descrizione_lunga = f"{intro}. {nome} con {tomaia}, {suola}. {finale}"
    descrizione_breve = f"{categoria} in {materiale} con {random.choice(dettagli)}. Suola in {random.choice(suole)}."

    return pd.Series([descrizione_lunga, descrizione_breve])

uploaded_file = st.file_uploader("Carica il file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
    columns = df.columns.tolist()

    name_col = st.selectbox("Colonna nome prodotto", options=columns, index=columns.index("name") if "name" in columns else 0)
    material_col = st.selectbox("Colonna materiale", options=columns, index=columns.index("material") if "material" in columns else 0)
    category_col = st.selectbox("Colonna categoria", options=columns, index=columns.index("category") if "category" in columns else 0)

    output_long = st.text_input("Nome colonna per la descrizione lunga", value="description")
    output_short = st.text_input("Nome colonna per la descrizione breve", value="short_description")

    if st.button("Genera Descrizioni"):
        df[[output_long, output_short]] = df.apply(generate_descriptions, axis=1, args=(name_col, material_col, category_col))
        st.success("Descrizioni generate con successo!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig')
        st.download_button(
            label="📥 Scarica il CSV con le descrizioni",
            data=csv,
            file_name='descrizioni_calzature.csv',
            mime='text/csv'
        )
