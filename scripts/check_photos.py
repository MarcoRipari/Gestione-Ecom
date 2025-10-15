import asyncio
import aiohttp
import os
import json
import gspread
import io
import hashlib
import imagehash
from PIL import Image, ImageChops
from datetime import datetime
from typing import List, Dict
from google.oauth2.service_account import Credentials
import dropbox
from dropbox.files import WriteMode

DEBUG = 0  # Imposta a 0 per eseguire lo script completo
async def test_debug():
    sku = "2012889010C02"
    riscattare = True
    sem = asyncio.Semaphore(1)

    async with aiohttp.ClientSession() as session:
        #url = "https://drive.google.com/uc?export=download&id=1GVjmhKEPNQTiFfnAuDM-NsCJEC2DLBM2"
        url = "https://drive.google.com/uc?export=download&id=1lLyYfJmbRxR7aRVdKdt1Ev78SUHALi0d"

        async def fetch_image():
            try:
                async with session.get(url, timeout=TIMEOUT_SECONDS) as get_resp:
                    if get_resp.status == 200:
                        img_bytes = await get_resp.read()
                        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    else:
                        print(f"❌ Impossibile scaricare immagine per {sku}. Status: {get_resp.status}")
                        return None
            except Exception as e:
                print(f"❌ Errore durante il download immagine: {e}")
                return None

        new_img = await fetch_image()
        if not new_img:
            return

        old_name, old_img = get_dropbox_latest_image(sku)
        if riscattare:
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
                        print(f"📦 Vecchia immagine rinominata in {new_old_name}")
                    except Exception as e:
                        print(f"⚠️ Errore nel rinominare {old_name}: {e}")
                save_image_to_dropbox(sku, f"{sku}.jpg", new_img)
                print(f"✅ Nuova immagine salvata per {sku}")
            else:
                print(f"🟰 Immagine identica già presente per {sku}. Nessuna azione.")
        else:
            print(f"⚠️ RISCATTARE non era True, nessuna azione eseguita.")
# -------------------------------
# CONFIG
# -------------------------------
SHEET_ID = os.environ.get("FOTO_GSHEET_ID")
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
DROPBOX_TOKEN = os.environ.get("DROPBOX_TOKEN")
DROPBOX_REFRESH_TOKEN = os.environ.get("DROPBOX_REFRESH_TOKEN")
DROPBOX_APP_KEY = os.environ.get("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.environ.get("DROPBOX_APP_SECRET")

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
#dbx = dropbox.Dropbox(DROPBOX_TOKEN)
dbx = dropbox.Dropbox(
    oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
    app_key=DROPBOX_APP_KEY,
    app_secret=DROPBOX_APP_SECRET
)

# -------------------------------
# UTILS
# -------------------------------
def get_sheet(sheet_id, tab_name):
    return gs_client.open_by_key(sheet_id).worksheet(tab_name)

def hash_image(image: Image.Image) -> str:
    """Genera hash MD5 per un'immagine."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    return hashlib.md5(img_bytes.getvalue()).hexdigest()

def images_are_equal(img1: Image.Image, img2: Image.Image, threshold: int = 0) -> bool:
    """Confronta le immagini usando perceptual hash (pHash)."""
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return hash1 - hash2 <= threshold  # soglia 0 = identiche, 1-2 = molto simili

def test_image_are_equal(img1: Image.Image, img2: Image.Image, threshold: int = 0) -> bool:
    """Confronta le immagini usando perceptual hash (pHash)."""
    hash1 = imagehash.phash(img1)
    hash2 = imagehash.phash(img2)
    return hash1, hash2, hash1 - hash2 <= threshold  # soglia 0 = identiche, 1-2 = molto simili

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

    dbx.files_upload(img_bytes.read(), file_path, mode=WriteMode("overwrite"))

async def check_photo(sku: str, riscattare: bool, sem: asyncio.Semaphore, session: aiohttp.ClientSession) -> (str, bool, bool):
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    async with sem:
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS, allow_redirects=True) as get_resp:
                if get_resp.status == 200:
                    img_bytes = await get_resp.read()
                    new_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    foto_salvata = False

                    if riscattare:
                        old_name, old_img = get_dropbox_latest_image(sku)
                        h1,h2,hdiff = test_images_are_equal(new_img, old_img)
                        print("DEBUG: old_img is None?", old_img is None)
                        print("DEBUG: hash old:", h1)
                        print("DEBUG: hash new:", h2)
                        print("DEBUG: pHash diff:", hdiff)

                        
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
                                    print(f"⚠️ Errore rinominando {old_name}: {e}")
                            save_image_to_dropbox(sku, f"{sku}.jpg", new_img)
                            foto_salvata = True

                    return sku, False, foto_salvata
                else:
                    return sku, True, False
        except Exception as e:
            print(f"❌ Errore fetch immagine {sku}: {e}")
            return sku, True, False

async def process_skus(data_rows: List[List[str]], sku_idx: int, riscattare_idx: int) -> Dict[str, bool]:
    results = {}
    foto_salvate = 0
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = {}
        for i, row in enumerate(data_rows):
            if len(row) > max(sku_idx, riscattare_idx):
                sku = row[sku_idx].strip()
                riscattare = row[riscattare_idx].strip().lower() == "true"
                if sku:
                    tasks[sku] = asyncio.create_task(check_photo(sku, riscattare, sem, session))
        for sku, task in tasks.items():
            try:
                sku, mancante, salvata = await task
                results[sku] = (mancante, salvata)
                if salvata:
                    foto_salvate += 1    
            except:
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
        consegnata_idx = header.index("CONSEGNATA")
    except ValueError as e:
        print(f"❌ Colonna mancante: {e}")
        return

    print(f"🔍 SKU totali: {len(rows)}")
    results, tot_foto_salvate = await retry_until_complete(rows, sku_idx, riscattare_idx)
    print(f"✅ Verificate: {len(results)}")

    output_col_k = []  # Colonna SCATTARE
    output_col_p = []  # Colonna RISCATTARE
    
    for i, row in enumerate(rows):
        sku = row[sku_idx].strip() if len(row) > sku_idx else ""
        mancante, salvata = results.get(sku, (None, False))
        output_col_k.append([str(mancante)])
    
        # Imposta RISCATTARE = FALSE solo se l'immagine è stata salvata effettivamente
        if salvata and len(row) > riscattare_idx:
            if row[riscattare_idx].strip().lower() == "true":
                if row[consegnata_idx].strip().lower() == "true":
                    output_col_p.append(["Check"])
                else:
                    output_col_p.append(["True"])
            elif row[riscattare_idx].strip().lower() == "check":
                output_col_p.append([""])
        else:
            if row[riscattare_idx].strip().lower() == "true":
                if row[consegnata_idx].strip().lower() == "true":
                    output_col_p.append(["Check"])
                else:
                    output_col_p.append(["True"])
            elif row[riscattare_idx].strip().lower() == "check":
                output_col_p.append(["Check"])
            else:
                output_col_p.append([""])
    
    # Aggiorna le due colonne nel foglio
    sheet.batch_update([
        {
            "range": f"K3:K{len(output_col_k)+2}",
            "values": output_col_k
        },
        {
            "range": f"P3:P{len(output_col_p)+2}",
            "values": output_col_p
        }
    ])
    
    print("✅ Google Sheet aggiornato")
    print(f"📦 Aggiornate {len(results)} SKU")
    print(f"🖼️ Foto scaricate su Dropbox: {tot_foto_salvate}")
    
if __name__ == "__main__":
    import asyncio
    if DEBUG:
        asyncio.run(test_debug())
    else:
        asyncio.run(main())
