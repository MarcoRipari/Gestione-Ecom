import os
import json
import aiohttp
import asyncio
import gspread
from google.oauth2.service_account import Credentials

async def check_single_photo(session, sku: str) -> tuple[str, bool]:
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    try:
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            return sku, response.status != 200
    except:
        return sku, True

async def main():
    creds_info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(os.environ["FOTO_GSHEET_ID"]).worksheet("LISTA")

    all_rows = sheet.get_all_values()
    headers, data = all_rows[:2], all_rows[2:]
    skus = [row[0] for row in data if row[0] and (len(row) < 11 or row[10].strip().lower() != "false")]

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*(check_single_photo(session, sku) for sku in skus))

    sku_to_result = {sku: str(missing) for sku, missing in results}

    values_to_update = [[sku_to_result.get(row[0], "True")] for row in data]
    sheet.update(values=values_to_update, range_name=f"K3:K{len(values_to_update)+2}")

if __name__ == "__main__":
    asyncio.run(main())
