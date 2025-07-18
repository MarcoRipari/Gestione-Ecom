# gdrive.py
import pandas as pd
import gspread
from google.oauth2 import service_account
import streamlit as st

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["GCP_SERVICE_ACCOUNT"],
    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
)
gsheet_client = gspread.authorize(credentials)

def get_sheet(sheet_id, tab):
    spreadsheet = gsheet_client.open_by_key(sheet_id)
    worksheets = spreadsheet.worksheets()
    for ws in worksheets:
        if ws.title.strip().lower() == tab.strip().lower():
            return ws
    return spreadsheet.add_worksheet(title=tab, rows="10000", cols="50")

def append_log(sheet_id, log_data):
    sheet = get_sheet(sheet_id, "logs")
    sheet.append_row(list(log_data.values()), value_input_option="RAW")

def overwrite_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    sheet.clear()
    df = df.fillna("").astype(str)
    data = [df.columns.tolist()] + df.values.tolist()
    sheet.update(data)

def append_to_sheet(sheet_id, tab, df):
    sheet = get_sheet(sheet_id, tab)
    df = df.fillna("").astype(str)
    values = df.values.tolist()
    sheet.append_rows(values, value_input_option="RAW")
