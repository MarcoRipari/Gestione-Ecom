# execution.py

import time
import json
import pandas as pd
from io import BytesIO
from typing import List, Dict, Tuple
from prompting import build_unified_prompt, get_blip_caption
from retrieval import retrieve_similar
from gdrive import append_to_sheet, append_log
from generation import generate_all_prompts


def check_existing_rows(df_input: pd.DataFrame, selected_langs: List[str], sheet_id: str, gsheet_client) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[dict]], List[int]]:
    existing_data = {}
    already_generated = {lang: [] for lang in selected_langs}
    rows_to_generate = []

    for lang in selected_langs:
        try:
            tab_df = pd.DataFrame(gsheet_client.open_by_key(sheet_id).worksheet(lang).get_all_records())
            tab_df = tab_df[["SKU", "Description", "Description2"]].dropna(subset=["SKU"])
            tab_df["SKU"] = tab_df["SKU"].astype(str)
            existing_data[lang] = tab_df.set_index("SKU")
        except:
            existing_data[lang] = pd.DataFrame(columns=["Description", "Description2"])

    for i, row in df_input.iterrows():
        sku = str(row.get("SKU", "")).strip()
        if not sku:
            rows_to_generate.append(i)
            continue

        all_present = True
        for lang in selected_langs:
            df_lang = existing_data.get(lang)
            if df_lang is None or sku not in df_lang.index:
                all_present = False
                break
            desc = df_lang.loc[sku]
            if not desc["Description"] or not desc["Description2"]:
                all_present = False
                break

        if all_present:
            for lang in selected_langs:
                desc = existing_data[lang].loc[sku]
                output_row = row.to_dict()
                output_row["Description"] = desc["Description"]
                output_row["Description2"] = desc["Description2"]
                already_generated[lang].append(output_row)
        else:
            rows_to_generate.append(i)

    return already_generated, rows_to_generate


def build_prompts_for_rows(df: pd.DataFrame, selected_langs: List[str], col_display_names: Dict[str, str],
                           k_simili: int, use_image: bool, faiss_index: Tuple, get_blip_caption, retrieve_similar,
                           desc_lunga_length, desc_breve_length, selected_tones) -> List[str]:
    prompts = []
    for _, row in df.iterrows():
        simili = retrieve_similar(row, faiss_index[1], faiss_index[0], k=k_simili) if k_simili > 0 else pd.DataFrame([])
        caption = get_blip_caption(row.get("Image 1", "")) if use_image and row.get("Image 1", "") else None
        prompt = build_unified_prompt(
            row=row,
            col_display_names=col_display_names,
            selected_langs=selected_langs,
            image_caption=caption,
            simili=simili,
            desc_lunga_length=desc_lunga_length,
            desc_breve_length=desc_breve_length,
            selected_tones=selected_tones
        )
        prompts.append(prompt)
    return prompts


def parse_generation_results(results: Dict[int, dict], df_input_to_generate: pd.DataFrame,
                             selected_langs: List[str], all_prompts: List[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    all_outputs = {lang: [] for lang in selected_langs}
    logs = []

    for i, (_, row) in enumerate(df_input_to_generate.iterrows()):
        result = results.get(i, {})
        sku = row.get("SKU", "")
        if "error" in result:
            logs.append({
                "sku": sku,
                "status": f"Errore: {result['error']}",
                "prompt": all_prompts[i],
                "output": "",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            continue

        for lang in selected_langs:
            lang_data = result.get("result", {}).get(lang.lower(), {})
            descr_lunga = lang_data.get("desc_lunga", "").strip()
            descr_breve = lang_data.get("desc_breve", "").strip()

            output_row = row.to_dict()
            output_row["Description"] = descr_lunga
            output_row["Description2"] = descr_breve
            all_outputs[lang].append(output_row)

        log_entry = {
            "sku": sku,
            "status": "OK",
            "prompt": all_prompts[i],
            "output": json.dumps(result["result"], ensure_ascii=False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if "usage" in result:
            usage = result["usage"]
            log_entry.update({
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "estimated_cost_usd": round(usage.get("total_tokens", 0) / 1000 * 0.001, 6)
            })

        logs.append(log_entry)

    return all_outputs, logs


def process_generation(df_input: pd.DataFrame, selected_langs: List[str], sheet_id: str, gsheet_client,
                       col_display_names: Dict[str, str], faiss_index: Tuple, k_simili: int, use_image: bool,
                       desc_lunga_length: int, desc_breve_length: int, selected_tones: List[str]) -> Tuple[Dict[str, List[dict]], List[dict]]:
    
    # ✅ Recupero righe già esistenti
    already_generated, rows_to_generate = check_existing_rows(df_input, selected_langs, sheet_id, gsheet_client)
    df_input_to_generate = df_input.iloc[rows_to_generate]

    # ✅ Prompt generation
    prompts = build_prompts_for_rows(df_input_to_generate, selected_langs, col_display_names, k_simili,
                                     use_image, faiss_index, get_blip_caption, retrieve_similar,
                                     desc_lunga_length, desc_breve_length, selected_tones)

    results = asyncio.run(generate_all_prompts(prompts))
    new_outputs, logs = parse_generation_results(results, df_input_to_generate, selected_langs, prompts)

    # ✅ Merge esistenti + nuovi
    for lang in selected_langs:
        already_generated[lang].extend(new_outputs[lang])

    return already_generated, logs
