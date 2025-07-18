from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import streamlit as st

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
    except Exception:
        return ""
