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
        pass

    try:
        dbx.files_upload(img_bytes.read(), file_path, mode=WriteMode("overwrite"))
        try:
            shared_link = dbx.sharing_create_shared_link_with_settings(file_path)
            print(f"📸 {sku} salvata → {shared_link.url.replace('?dl=0', '?raw=1')}")
        except Exception as e:
            print(f"⚠️ Errore link condiviso {sku}: {e}")
    except Exception as e:
        print(f"❌ Errore upload Dropbox per {sku}: {e}")

async def check_photo(sku: str, riscattare: bool, sem: asyncio.Semaphore, session: aiohttp.ClientSession) -> (str, bool, bool):
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    async with sem:
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS, allow_redirects=True) as get_resp:
                if get_resp.status == 200:
                    img_bytes = await get_resp.read()
                    try:
                        new_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    except Exception as e:
                        print(f"❌ Errore apertura immagine {sku}: {e}")
                        return sku, True, False

                    foto_salvata = False
                    if riscattare:
                        old_name, old_img = get_dropbox_latest_image(sku)
                        if not old_img or not images_are_equal(new_img, old_img):
                            if old_name:
                                date_suffix = datetime.now().strftime("%d%m%Y")
                                ext = old_name.split(".")[-1]
                                new_old_name = f"{sku}_{date_suffix}.{ext}"
                                try:
                                    dbx.files_move_v2(
                                        from_path=f"/repository/{sku}/{old_name}",
                                        to_path=f"/repository/{sku}/{new_old_name}",
                                        allow_shared_folder=True,
                                        autorename=True
                                    )
                                except Exception as e:
                                    print(f"⚠️ Errore rinomina {sku}: {e}")
                            save_image_to_dropbox(sku, f"{sku}.jpg", new_img)
                            foto_salvata = True
                    return sku, False, foto_salvata
                else:
                    return sku, True, False
        except Exception as e:
            import traceback
            print(f"❌ Errore fetch immagine {sku}: {str(e)}")
            traceback.print_exc()
            return sku, True, False

async def process_skus(data_rows: List[List[str]], sku_idx: int, riscattare_idx: int) -> Dict[str, bool]:
    results = {}
    foto_salvate = 0
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
                sku, mancante, salvata = await task
                results[sku] = mancante
                if salvata:
                    foto_salvate += 1
            except Exception as e:
                print(f"❌ Errore task {sku}: {e}")
                results[sku] = True
    return results, foto_salvate

async def retry_until_complete(data_rows, sku_idx, riscattare_idx) -> (Dict[str, bool], int):
    checked = {}
    foto_salvate_totali = 0
    retries = 0
    while retries < RETRY_LIMIT:
        pending = [row for row in data_rows if row[sku_idx].strip() not in checked]
        if not pending:
            break
        partial, salvate = await process_skus(pending, sku_idx, riscattare_idx)
        checked.update(partial)
        foto_salvate_totali += salvate
        retries += 1
    return checked, foto_salvate_totali

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

    results, tot_foto_salvate = await retry_until_complete(rows, sku_idx, riscattare_idx)

    output_column = []
    for row in rows:
        sku = row[sku_idx].strip() if len(row) > sku_idx else ""
        output_column.append([str(results.get(sku, ""))])

    sheet.update(
        range_name=f"K3:K{len(output_column)+2}",
        values=output_column,
        value_input_option="RAW"
    )

    print("✅ Google Sheet aggiornato")
    print(f"📦 Aggiornate {len(results)} SKU")
    print(f"🖼️ Foto salvate su Dropbox: {tot_foto_salvate}")

if __name__ == "__main__":
    asyncio.run(main())
