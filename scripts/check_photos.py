import asyncio
import aiohttp
import os
import json
import gspread
import io
from PIL import Image
from datetime import datetime
from typing import List, Dict
from google.oauth2.service_account import Credentials
import dropbox
from dropbox.files import WriteMode

# -------------------------------
# CONFIG
# -------------------------------
SHEET_ID = os.environ.get("FOTO_GSHEET_ID")
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
DROPBOX_TOKEN = os.environ.get("DROPBOX_TOKEN")

FOGLIO = "LISTA"
MAX_CONCURRENT = 40
RETRY_LIMIT = 3
TIMEOUT_SECONDS = 10

# -------------------------------
# AUTENTICAZIONE
# -------------------------------
credentials = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gs_client = gspread.authorize(credentials)
dbx = dropbox.Dropbox(DROPBOX_TOKEN)

# -------------------------------
# UTILS
# -------------------------------
def get_sheet(sheet_id, tab_name):
    return gs_client.open_by_key(sheet_id).worksheet(tab_name)

def ensure_dropbox_folder(path: str):
    try:
        dbx.files_get_metadata(path)
    except dropbox.exceptions.ApiError as e:
        if isinstance(e.error, dropbox.files.GetMetadataError) or "not_found" in str(e).lower():
            dbx.files_create_folder_v2(path)

def images_are_equal(img1: Image.Image, img2: Image.Image) -> bool:
    return list(img1.getdata()) == list(img2.getdata())

def get_dropbox_latest_image(sku: str) -> (str, Image.Image):
    folder_path = f"/repository/{sku}"
    try:
        res = dbx.files_list_folder(folder_path)
        jpgs = sorted(
            [entry for entry in res.entries if entry.name.lower().endswith(".jpg")],
            key=lambda e: e.client_modified,
            reverse=True
        )
        if not jpgs:
            return None, None
        latest = jpgs[0]
        _, resp = dbx.files_download(latest.path_display)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return latest.name, img
    except dropbox.exceptions.ApiError:
        return None, None

def save_image_to_dropbox(sku: str, filename: str, image: Image.Image):
    folder_path = f"/repository/{sku}"
    file_path = f"{folder_path}/{filename}"

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    try:
        dbx.files_create_folder_v2(folder_path)
    except dropbox.exceptions.ApiError:
        pass  # La cartella esiste già

    dbx.files_upload(img_bytes.read(), file_path, mode=WriteMode("overwrite"))

    # ✅ Ottieni link condivisibile
    try:
        shared_link = dbx.sharing_create_shared_link_with_settings(file_path)
        print(f"✅ Foto salvata: {file_path}")
        print(f"🔗 Link: {shared_link.url.replace('?dl=0', '?raw=1')}")
    except dropbox.exceptions.ApiError as e:
        print(f"⚠️ Impossibile creare link condiviso per {file_path}: {e}")

async def check_photo(sku: str, riscattare: bool, sem: asyncio.Semaphore, session: aiohttp.ClientSession) -> (str, bool):
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    async with sem:
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    img_bytes = await response.read()
                    new_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    if riscattare:
                        folder_path = f"/repository/{sku}"
                        ensure_dropbox_folder(folder_path)

                        old_name, old_img = get_dropbox_latest_image(sku)
                        if old_img and images_are_equal(new_img, old_img):
                            return sku, False  # Nessuna azione

                        # Rinominare vecchia immagine
                        if old_name:
                            date_suffix = datetime.now().strftime("%d%m%Y")
                            ext = old_name.split(".")[-1]
                            new_old_name = f"{sku}_{date_suffix}.{ext}"
                            dbx.files_move_v2(
                                from_path=f"{folder_path}/{old_name}",
                                to_path=f"{folder_path}/{new_old_name}",
                                allow_shared_folder=True,
                                autorename=True
                            )

                        save_image_to_dropbox(sku, f"{sku}.jpg", new_img)

                    # restituisce True solo se la foto è realmente mancante
                    missing = response.status != 200
                    return sku, missing
                else:
                    return sku, True  # Foto mancante
        except:
            return sku, True

async def process_skus(data_rows: List[List[str]], sku_idx: int, riscattare_idx: int) -> Dict[str, bool]:
    results = {}
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = {}
        for row in data_rows:
            if len(row) > max(sku_idx, riscattare_idx):
                sku = row[sku_idx].strip()
                riscattare = row[riscattare_idx].strip().lower() == "true"
                if sku:
                    tasks[sku] = asyncio.create_task(check_photo(sku, riscattare, sem, session))
        for sku, task in tasks.items():
            try:
                result = await task
                results[result[0]] = result[1]
            except:
                results[sku] = True
    return results

async def retry_until_complete(data_rows, sku_idx, riscattare_idx) -> Dict[str, bool]:
    checked = {}
    retries = 0
    while retries < RETRY_LIMIT:
        pending = [row for row in data_rows if row[sku_idx].strip() not in checked]
        if not pending:
            break
        print(f"🔁 Retry {retries+1}: {len(pending)} SKU")
        partial = await process_skus(pending, sku_idx, riscattare_idx)
        checked.update(partial)
        retries += 1
    return checked

# -------------------------------
# MAIN
# -------------------------------
async def main():
    sheet = get_sheet(SHEET_ID, FOGLIO)
    all_data = sheet.get_all_values()
    if len(all_data) < 3:
        print("❌ Foglio vuoto.")
        return

    header = all_data[1]
    rows = all_data[2:]

    try:
        sku_idx = header.index("SKU")
        riscattare_idx = header.index("RISCATTARE")
    except ValueError as e:
        print(f"❌ Colonna mancante: {e}")
        return

    print(f"🔍 SKU totali: {len(rows)}")
    results = await retry_until_complete(rows, sku_idx, riscattare_idx)
    print(f"✅ Verificate: {len(results)}")
    
    # ✅ Log per le prime 10 SKU
    print("📋 Log esistenza foto per le prime 10 SKU:")
    for i, (sku, exists) in enumerate(results.items()):
        print(f"  {sku}: {'MANCANTE' if exists else 'ESISTE'}")
        if i >= 9:
            break

    output_column = []
    for row in rows:
        sku = row[sku_idx].strip() if len(row) > sku_idx else ""
        if sku in results:
            output_column.append([str(results[sku])])  # True solo se mancante
        else:
            output_column.append([""])

    sheet.update(
        range_name=f"K3:K{len(output_column)+2}",
        values=output_column,
        value_input_option="RAW"
    )
    print("✅ Google Sheet aggiornato")

if __name__ == "__main__":
    asyncio.run(main())
