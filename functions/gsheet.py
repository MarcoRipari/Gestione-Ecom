import streamlit as st
from google.oauth2 import service_account
import gspread
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.http import MediaInMemoryUpload
from gspread.utils import rowcol_to_a1
from gspread_formatting import CellFormat, NumberFormat, format_cell_ranges
import gspread.utils
from googleapiclient.discovery import build
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ---------------------------
# 📊 Google Sheets
# ---------------------------
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
)
gsheet_client = gspread.authorize(credentials)
drive_service = build('drive', 'v3', credentials=credentials)

def get_sheet(sheet_id, tab):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    worksheets = spreadsheet.worksheets()
    
    # Confronto case-insensitive per maggiore robustezza
    for ws in worksheets:
        if ws.title.strip().lower() == tab.strip().lower():
            return ws

    # Se non trovato, lo crea
    return spreadsheet.add_worksheet(title=tab, rows="10000", cols="50")

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    sheet.append_rows(values, value_input_option="RAW")  # ✅ chiamata unica

def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")
