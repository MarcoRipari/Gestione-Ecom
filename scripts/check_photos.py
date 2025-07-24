import asyncio
import aiohttp
import os
import json
import gspread
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from PIL import Image
from typing import List, Dict

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
SHEET_ID = os.environ.get("FOTO_GSHEET_ID")
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
REPOSITORY_FOLDER_ID = os.environ.get("REPOSITORY_FOLDER_ID")

FOGLIO = "LISTA"
MAX_CONCURRENT = 40
RETRY_LIMIT = 3
TIMEOUT_SECONDS = 10

# -------------------------------
# AUTENTICAZIONE GOOGLE
# -------------------------------
credentials = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
gs_client = gspread.authorize(credentials)
drive_service = build("drive", "v3", credentials=credentials)

# -------------------------------
# UTILS
# -------------------------------
def get_sheet(sheet_id, tab_name):
    spreadsheet = gs_client.open_by_key(sheet_id)
    return spreadsheet.worksheet(tab_name)

def get_drive_file(folder_id, filename):
    results = drive_service.files().list(
        q=f"name='{filename}' and '{folder_id}' in parents and trashed = false",
        spaces='drive',
        fields="files(id, name)"
    ).execute()
    files = results.get("files", [])
    return files[0] if files else None

def create_folder_if_not_exists(parent_id, folder_name):
    folder = get_drive_file(parent_id, folder_name)
    if folder:
        return folder["id"]
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id]
    }
    folder = drive_service.files().create(body=file_metadata, fields="id").execute()
    return folder["id"]

def download_drive_image(file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return Image.open(fh)

def images_are_equal(img1: Image.Image, img2: Image.Image) -> bool:
    return list(img1.getdata()) == list(img2.getdata())

async def check_photo(sku: str, riscattare: bool, sem: asyncio.Semaphore, session: aiohttp.ClientSession) -> (str, bool):
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    async with sem:
        try:
            async with session.get(url, timeout=TIMEOUT_SECONDS) as response:
                if response.status == 200:
                    img_bytes = await response.read()
                    if riscattare:
                        folder_id = create_folder_if_not_exists(REPOSITORY_FOLDER_ID, sku)
                        filename = f"{sku}.jpg"
                        existing_file = get_drive_file(folder_id, filename)

                        new_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                        if existing_file:
                            existing_img = download_drive_image(existing_file["id"]).convert("RGB")
                            if images_are_equal(existing_img, new_image):
                                return sku, False  # immagine già presente e uguale
                            else:
                                # elimina e aggiorna
                                drive_service.files().delete(fileId=existing_file["id"]).execute()

                        # salva nuova immagine
                        fh = io.BytesIO()
                        new_image.save(fh, format="JPEG")
                        fh.seek(0)
                        media = MediaIoBaseUpload(fh, mimetype="image/jpeg")
                        drive_service.files().create(
                            body={"name": filename, "parents": [folder_id]},
                            media_body=media,
                            fields="id"
                        ).execute()
                    return sku, False  # esiste
                else:
                    return sku, True  # mancante
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
        pending_rows = [
            row for row in data_rows
            if row[sku_idx].strip() not in checked
        ]
        if not pending_rows:
            break
        print(f"🔁 Retry {retries+1}, checking {len(pending_rows)} SKU...")
        partial = await process_skus(pending_rows, sku_idx, riscattare_idx)
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
        print("❌ Foglio vuoto o troppo corto.")
        return

    header = all_data[1]
    rows = all_data[2:]

    try:
        sku_idx = header.index("SKU")
        riscattare_idx = header.index("RISCATTARE")
    except ValueError as e:
        print(f"❌ Colonna mancante: {e}")
        return

    print(f"🔍 Totale righe da analizzare: {len(rows)}")

    results = await retry_until_complete(rows, sku_idx, riscattare_idx)

    print(f"✅ Controllate: {len(results)} SKU")

    output_column = []
    for row in rows:
        sku = row[sku_idx].strip() if len(row) > sku_idx else ""
        if sku in results:
            output_column.append([str(results[sku])])
        else:
            output_column.append([""])

    range_k = f"K3:K{len(output_column)+2}"
    print(f"✍️ Aggiorno Google Sheet su {range_k}")
    sheet.update(values=output_column, range_name=range_k, value_input_option="RAW")
    print("✅ Operazione completata.")

if __name__ == "__main__":
    asyncio.run(main())
