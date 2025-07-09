import streamlit as st
import pandas as pd
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import tempfile
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG ===
OPENAI_API_KEY = "sk-..."
SHEET_ID = "1R9diynXe..."
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

client = OpenAI(api_key=OPENAI_API_KEY)

# === GOOGLE SHEET HELPER ===
def connect_to_gsheet(creds, sheet_id):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        json.dump(creds, tmp); tmp.flush()
        credentials = ServiceAccountCredentials.from_json_keyfile_name(tmp.name, scope)
    gc = gspread.authorize(credentials)
    return gc.open_by_key(sheet_id).sheet1

# === RAG: trova esempi simili ===
def get_similar(df_hist, row, k=2):
    if df_hist.empty: return []
    df_hist = df_hist.dropna(subset=["description", "short_description"])
    df_hist["txt"] = df_hist.apply(lambda r: " ".join(str(r[c]) for c in df_hist.columns if c not in ["description","short_description"]), axis=1)
    query_txt = " ".join(str(v) for k,v in row.items() if pd.notna(v) and k not in ["description","short_description"])
    vect = TfidfVectorizer().fit(df_hist["txt"].tolist() + [query_txt])
    qv = vect.transform([query_txt])
    cv = vect.transform(df_hist["txt"])
    sims = cosine_similarity(qv, cv).flatten()
    ix = sims.argsort()[-k:][::-1]
    return df_hist.iloc[ix][["description", "short_description"]].values.tolist()

# === FUNZIONE GENERAZIONE JSON + RAG ===
def generate_with_rag(row, examples):
    prod = ", ".join(f"{k}: {v}" for k,v in row.items() if pd.notna(v) and k not in ["description","short_description"])
    exs = "\n".join(f'  "{i+1}": {{"descrizione_lunga": "{ex[0]}", "descrizione_corta": "{ex[1]}"}}'
                    for i,ex in enumerate(examples))
    prompt = f"""
Fornisci output **solo in JSON**:

{{
  "examples": {{
{exs}
  }},
  "new": {{
    "product_info": "{prod}"
  }}
}}

Dopo i \"examples\" genera:
{{"descrizione_lunga": "...", "descrizione_corta": "..."}}
"""
    resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}], temperature=0.7)
    txt = resp.choices[0].message.content.strip()
    try:
        j = json.loads(txt)
        nd = j.get("descrizione_lunga","") or "Descrizione lunga non trovata"
        ns = j.get("descrizione_corta","") or "Descrizione corta non trovata"
    except:
        nd, ns = "Errore parsing JSON", "Errore parsing JSON"
    return nd, ns

# === UPLOAD FILE ===
uploaded = st.file_uploader("📤 Carica CSV", type="csv")
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.dataframe(df.head())

# Column setup
if "description" not in df: df["description"] = ""
if "short_description" not in df: df["short_description"] = ""

# Stima costo/token
n = len(df)
tok = sum(len(str(r))//4 for _,r in df.iterrows())
cost = tok/1000 * 0.0015
st.info(f"🔢 Token stimati: {tok} | 💰 Costo stimato: ${cost:.4f}")

if st.button("🚀 Genera con RAG & JSON"):
    sheet = connect_to_gsheet(CREDENTIALS_JSON, SHEET_ID)
    hist = pd.DataFrame(sheet.get_all_records())
    rows = []
    progress = st.progress(0)
    for i, row in df.iterrows():
        exs = get_similar(hist, row, k=2)
        ld, sd = generate_with_rag(row, exs)
        r = row.to_dict()
        r["description"] = ld
        r["short_description"] = sd
        rows.append(r)
        progress.progress((i+1)/n)
    df2 = pd.DataFrame(rows)
    sheet.append_rows(df2.values.tolist())  # append
    st.success("✅ Descrizioni generate e aggiunte allo sheet!")
    st.download_button("📥 Scarica CSV", df2.to_csv(index=False).encode(), "out.csv", "text/csv")
