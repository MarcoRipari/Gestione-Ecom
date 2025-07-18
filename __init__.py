# Importazioni comode da usare altrove nel progetto
from .prompting import build_unified_prompt
from .gdrive import get_sheet, append_log, append_to_sheet, overwrite_sheet
from .retrieval import (
    load_model,
    build_faiss_index,
    retrieve_similar,
    estimate_embedding_time,
    benchmark_faiss,
)
from .execution import (
    generate_descriptions,
    async_generate_description,
    generate_all_prompts,
    calcola_tokens
)
