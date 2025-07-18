import json
from openai import AsyncOpenAI
from typing import List
import streamlit as st

LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco"
}

client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def build_unified_prompt(row, col_display_names, selected_langs, image_caption=None, simili=None):
    fields = []
    for col in col_display_names:
        if col in row and pd.notna(row[col]):
            label = col_display_names[col]
            fields.append(f"- {label}: {row[col]}")
    product_info = "\n".join(fields)

    lang_list = ", ".join([LANG_NAMES.get(lang, lang) for lang in selected_langs])
    image_line = f"\nAspetto visivo: {image_caption}" if image_caption else ""

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

    prompt = f"""Scrivi due descrizioni per una calzatura da vendere online (e-commerce) in ciascuna delle seguenti lingue: {lang_list}.

>>> FORMATO OUTPUT
{{"it":{{"desc_lunga":"...","desc_breve":"..."}}, "en":{{...}}, "fr":{{...}}, "de":{{...}}}}

>>> GUIDA STILE
- Tono: {", ".join(st.session_state.selected_tones)}
- Ometti sempre: Codice, Nome, Marca, Colore (nemmeno in forma implicita)
- Lingua: adatta al paese target
- Non usare il genere
- Usa sempre la parola strappo, niente sinonimi ne velcro
- Evita le percentuali materiali

>>> REGOLE
- desc_lunga: {st.session_state.desc_lunga_length} parole → enfasi su comfort, materiali, utilizzo
- desc_breve: {st.session_state.desc_breve_length} parole → adatta a social media o schede prodotto rapide

>>> INFO ARTICOLO
{product_info}
{image_line}

{sim_text}
"""
    return prompt

def generate_descriptions(prompt):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=3000
    )
    return response.choices[0].message.content

async def async_generate_description(prompt: str, idx: int):
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000
        )
        content = response.choices[0].message.content
        usage = response.usage
        data = json.loads(content)
        return idx, {"result": data, "usage": usage.model_dump()}
    except Exception as e:
        return idx, {"error": str(e)}

async def generate_all_prompts(prompts: List[str]) -> dict:
    tasks = [async_generate_description(prompt, idx) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)
