import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pandas as pd

LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco"
}
LANG_LABELS = {v.capitalize(): k for k, v in LANG_NAMES.items()}

selected_tones = []
desc_lunga_length = 0
desc_breve_length = 0

# ---------------------------
# 📦 Embedding & FAISS Setup
# ---------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=st.secrets["HF_TOKEN"])
    return model.to("cpu")

model = load_model()

def embed_texts(texts: List[str], batch_size=32) -> List[List[float]]:
    return model.encode(texts, show_progress_bar=False, batch_size=batch_size).tolist()

def hash_dataframe_and_weights(df: pd.DataFrame, col_weights: Dict[str, float]) -> str:
    df_bytes = pickle.dumps((df.fillna("").astype(str), col_weights))
    return hashlib.md5(df_bytes).hexdigest()

def build_faiss_index(df: pd.DataFrame, col_weights: Dict[str, float], cache_dir="faiss_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hash_dataframe_and_weights(df, col_weights)
    cache_path = os.path.join(cache_dir, f"{cache_key}.index")

    if os.path.exists(cache_path):
        index = faiss.read_index(cache_path)
        return index, df

    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    vectors = embed_texts(texts)
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, cache_path)

    return index, df

def retrieve_similar(query_row: pd.Series, df: pd.DataFrame, index, k=5, col_weights: Dict[str, float] = {}):
    parts = []
    for col in df.columns:
        if pd.notna(query_row[col]):
            weight = col_weights.get(col, 1)
            if weight > 0:
                parts.append((f"{col}: {query_row[col]} ") * int(weight))
    query_text = " ".join(parts)

    query_vector = embed_texts([query_text])[0]
    D, I = index.search(np.array([query_vector]).astype("float32"), k)

    # 🔍 DEBUG
    logging.info(f"QUERY TEXT: {query_text[:300]} ...")
    logging.info(f"INDICI trovati: {I[0]}")
    logging.info(f"Distanze: {D[0]}")
    
    return df.iloc[I[0]]

def estimate_embedding_time(df: pd.DataFrame, col_weights: Dict[str, float], sample_size: int = 10) -> float:
    """
    Stima il tempo totale per embeddare tutti i testi del dataframe.
    """
    texts = []
    for _, row in df.head(sample_size).iterrows():
        parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                weight = col_weights.get(col, 1)
                if weight > 0:
                    parts.append((f"{col}: {row[col]} ") * int(weight))
        texts.append(" ".join(parts))

    start = time.time()
    _ = embed_texts(texts)
    elapsed = time.time() - start
    avg_time_per_row = elapsed / sample_size
    total_estimated_time = avg_time_per_row * len(df)

    return total_estimated_time

def benchmark_faiss(df, col_weights, query_sample_size=10):
    import os

    st.markdown("### ⏱️ Benchmark FAISS + Embedding")

    start_embed = time.time()
    texts = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" * int(col_weights.get(col, 1))
                 for col in df.columns if pd.notna(row[col])]
        texts.append(" ".join(parts))
    vectors = embed_texts(texts)
    embed_time = time.time() - start_embed

    start_faiss = time.time()
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors).astype("float32"))
    faiss.write_index(index, "tmp_benchmark.index")
    index_time = time.time() - start_faiss

    index_size = os.path.getsize("tmp_benchmark.index")

    # Test query
    query_times = []
    for i in range(min(query_sample_size, len(df))):
        qtext = texts[i]
        start_q = time.time()
        _ = index.search(np.array([vectors[i]]).astype("float32"), 5)
        query_times.append(time.time() - start_q)

    avg_query_time = sum(query_times) / len(query_times)

    st.write({
        "🚀 Tempo embedding totale (s)": round(embed_time, 2),
        "📄 Tempo medio per riga (ms)": round(embed_time / len(df) * 1000, 2),
        "🏗️ Tempo costruzione FAISS (s)": round(index_time, 2),
        "💾 Dimensione index (KB)": round(index_size / 1024, 1),
        "🔍 Tempo medio query (ms)": round(avg_query_time * 1000, 2),
    })

    os.remove("tmp_benchmark.index")

# ---------------------------
# 🎨 Visual Embedding
# ---------------------------
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_auth_token=st.secrets["HF_TOKEN"]
    )
    return processor, model
    
def get_blip_caption(image_url: str) -> str:
    try:
        processor, model = load_blip_model()
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = processor(raw_image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        # st.warning(f"⚠️ Errore nel captioning: {str(e)}")
        return ""
        
# ---------------------------
# 🧠 Prompting e Generazione
# ---------------------------
def build_unified_prompt(row, col_display_names, selected_langs, image_caption=None, simili=None):
    # Costruzione scheda tecnica
    fields = []
    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"- {label}: {row[col]}")
    product_info = "\n".join(fields)

    # Elenco lingue in stringa
    lang_list = ", ".join([LANG_NAMES.get(lang, lang) for lang in selected_langs])

    # Caption immagine
    image_line = f"\nAspetto visivo: {image_caption}" if image_caption else ""

    # Descrizioni simili
    sim_text = ""
    if simili is not None and not simili.empty:
        sim_lines = []
        for _, ex in simili.iterrows():
            dl = ex.get("Description", "").strip()
            db = ex.get("Description2", "").strip()
            if dl and db:
                sim_lines.append(f"- {dl}\n  {db}")
        if sim_lines:
            sim_text = "\nDescrizioni simili:\n" + "\n".join(sim_lines)

    # Prompt finale
    prompt = f"""Scrivi due descrizioni per una calzatura da vendere online (e-commerce) in ciascuna delle seguenti lingue: {lang_list}.

>>> FORMATO OUTPUT
{{"it":{{"desc_lunga":"...","desc_breve":"..."}}, "en":{{...}}, "fr":{{...}}, "de":{{...}}}}

>>> GUIDA STILE
- Tono: {", ".join(selected_tones)}
- Ometti sempre: Codice, Nome, Marca, Colore (nemmeno in forma implicita)
- Lingua: adatta al paese target
- Non usare il genere
- Usa sempre la parola strappo, niente sinonimi ne velcro
- Evita le percentuali materiali

>>> REGOLE
- desc_lunga: {desc_lunga_length} parole → enfasi su comfort, materiali, utilizzo
- desc_breve: {desc_breve_length} parole → adatta a social media o schede prodotto rapide

>>> INFO ARTICOLO
{product_info}
{image_line}

{sim_text}
"""
    return prompt

def calcola_tokens(df_input, col_display_names, selected_langs, selected_tones, desc_lunga_length, desc_breve_length, k_simili, use_image, faiss_index, DEBUG=False):
    if df_input.empty:
        return None, None, "❌ Il CSV è vuoto"

    row = df_input.iloc[0]

    simili = pd.DataFrame([])
    if k_simili > 0 and faiss_index:
        index, index_df = faiss_index
        simili = retrieve_similar(row, index_df, index, k=k_simili, col_weights=st.session_state.col_weights)

    caption = get_blip_caption(row.get("Image 1", "")) if use_image and row.get("Image 1", "") else None

    prompt = build_unified_prompt(
        row=row,
        col_display_names=col_display_names,
        selected_langs=selected_langs,
        image_caption=caption,
        simili=simili
    )

    # Token estimation (~4 chars per token)
    num_chars = len(prompt)
    token_est = num_chars // 4
    cost_est = round(token_est / 1000 * 0.001, 6)

    if DEBUG:
        st.code(prompt)
        st.markdown(f"📊 **Prompt Length**: {num_chars} caratteri ≈ {token_est} token")
        st.markdown(f"💸 **Costo stimato per riga**: ${cost_est:.6f}")

    return token_est, cost_est, prompt
