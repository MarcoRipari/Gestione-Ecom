# generation.py

import json
from openai import AsyncOpenAI
import asyncio
import streamlit as st

client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

async def generate_all_prompts(prompts: list[str]) -> dict:
    tasks = [async_generate_description(prompt, idx) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return dict(results)
