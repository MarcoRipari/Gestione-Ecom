import asyncio
import aiohttp
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from typing import List, Dict

# -------------------------------
# CONFIGURAZIONE
# -------------------------------
SHEET_ID = os.environ.get("FOTO_GSHEET_ID")
SERVICE_ACCOUNT_JSON = os.environ.get("SERVICE_ACCOUNT_JSON")
FOGLIO = "LISTA"
MAX_CONCURRENT = 40
RETRY_LIMIT = 3
TIMEOUT_SECONDS = 10

# -------------------------------
# AUTENTICAZIONE GOOGLE
# -------------------------------
credentials = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
client = gspread.authorize(credentials)

# -------------------------------
# UTILS
# -------------------------------
def get_sheet(sheet_id, tab_name):
    spreadsheet = client.open_by_key(sheet_id)
    return spreadsheet.worksheet(tab_name)

async def check_photo(sku: str, sem: asyncio.Semaphore, session: aiohttp.ClientSession) -> bool:
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    async with sem:
        try:
            async with session.head(url, timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
                return response.status != 200  # True = manca
        except:
            return True  # considera mancante in caso di errore

async def process_skus(skus: List[str]) -> Dict[str, bool]:
    results = {}
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        tasks = {
            sku: asyncio.create_task(check_photo(sku, sem, session))
            for sku in skus
        }
        for sku, task in tasks.items():
            try:
                result = await task
                results[sku] = result
            except:
                results[sku] = True
    return results

async def retry_until_complete(skus: List[str]) -> Dict[str, bool]:
    checked = {}
    retries = 0
    while retries < RETRY_LIMIT:
        remaining = [sku for sku in skus if sku not in checked]
        if not remaining:
            break
        print(f"🔁 Retry {retries+1}, checking {len(remaining)} SKU...")
        partial = await process_skus(remaining)
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
        print("❌ Foglio vuoto o con meno di 3 righe.")
        return

    header = all_data[1]
    rows = all_data[2:]

    try:
        sku_idx = header.index("SKU")
    except ValueError:
        print("❌ Colonna SKU non trovata.")
        return

    sku_list = [row[sku_idx].strip() for row in rows if len(row) > sku_idx and row[sku_idx].strip()]

    print(f"🔎 Totale SKU da controllare: {len(sku_list)}")

    results = await retry_until_complete(sku_list)
    print(f"✅ Controllo completato: {len(results)} SKU verificate")

    # Prepara i risultati per la colonna K
    output_column = []
    for row in rows:
        sku = row[sku_idx].strip() if len(row) > sku_idx else ""
        if sku in results:
            output_column.append([str(results[sku])])
        else:
            output_column.append([""])

    start_row = 3
    end_row = start_row + len(output_column) - 1
    range_k = f"K{start_row}:K{end_row}"

    print(f"✍️ Scrittura risultati su {range_k}")
    sheet.update(values=output_column, range_name=range_k, value_input_option="RAW")

    print("✅ Scrittura completata")

if __name__ == "__main__":
    asyncio.run(main())
