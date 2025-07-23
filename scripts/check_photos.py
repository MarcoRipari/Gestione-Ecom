import asyncio
import aiohttp
import pandas as pd
import gspread
import os
import json
from google.oauth2.service_account import Credentials
import logging
from dotenv import load_dotenv

# 📦 Carica le variabili d'ambiente (es. .env)
load_dotenv()

# 📂 Configurazione logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔐 Credenziali e ID foglio
SERVICE_ACCOUNT_JSON = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
SHEET_ID = os.getenv("FOTO_GSHEET_ID")

# 🔓 Autenticazione Google
credentials = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON),
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gsheet_client = gspread.authorize(credentials)

def get_sheet(sheet_id, tab_name):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    for ws in spreadsheet.worksheets():
        if ws.title.lower() == tab_name.lower():
            return ws
    return spreadsheet.add_worksheet(title=tab_name, rows="10000", cols="50")

async def check_single_photo(session, sku: str) -> tuple[str, bool]:
    url = f"https://repository.falc.biz/fal001{sku.lower()}-1.jpg"
    try:
        async with session.head(url, timeout=10) as response:
            return sku, response.status != 200  # True = mancante
    except Exception as e:
        logging.warning(f"Errore su {sku}: {e}")
        return sku, True

async def controlla_foto_exist():
    logging.info("🔄 Caricamento dati da Google Sheets...")
    sheet_lista = get_sheet(SHEET_ID, "LISTA")
    rows = sheet_lista.get_all_values()

    if len(rows) < 3:
        logging.warning("⚠️ Nessuna riga da elaborare.")
        return

    headers = rows[1]
    data = rows[2:]
    df = pd.DataFrame(data, columns=headers)

    # ❌ Escludi righe con SCATTARE = False
    df = df[df["SCATTARE"].str.strip().str.lower() != "false"]
    df = df[df["SKU"].notna() & (df["SKU"].str.strip() != "")]

    sku_list = df["SKU"].tolist()
    logging.info(f"🔍 Verifica in corso su {len(sku_list)} SKU...")

    results = {}

    async with aiohttp.ClientSession() as session:
        tasks = [check_single_photo(session, sku) for sku in sku_list]
        responses = await asyncio.gather(*tasks)

    for sku, missing in responses:
        results[sku] = str(missing)  # Per compatibilità con Google Sheets (string)

    # 🔄 Scrittura risultati in colonna K (SCATTARE)
    logging.info("✍️ Scrittura risultati in Google Sheets...")
    update_values = []
    for _, row in df.iterrows():
        sku = row["SKU"]
        val = results.get(sku, "True")
        update_values.append([val])

    start_row = 3 + df.index.min()
    end_row = start_row + len(update_values) - 1
    range_name = f"K{start_row}:K{end_row}"

    sheet_lista.update(range_name=range_name, values=update_values, value_input_option="RAW")

    logging.info("✅ Controllo completato!")

if __name__ == "__main__":
    try:
        asyncio.run(controlla_foto_exist())
    except Exception as e:
        logging.error(f"Errore fatale: {e}")
