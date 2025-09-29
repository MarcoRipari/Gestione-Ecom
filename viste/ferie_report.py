import streamlit as st
import functions
import pandas as pd
ferie_sheet_id = st.secrets["FERIE_GSHEET_ID"]

def view()
    st.header("📅 Report ferie settimanale")
    
    # 1. Leggi dati ferie da GSheet
    sheet = functions.gsheet.get_sheet(ferie_sheet_id, "FERIE")
    ferie_data = sheet.get_all_values()
    ferie_df = pd.DataFrame(
        ferie_data[1:], columns=ferie_data[0]
    ) if len(ferie_data) > 1 else pd.DataFrame(columns=["NOME", "DATA INIZIO", "DATA FINE", "MOTIVO"])
    
    # 2. Selettore giorno iniziale
    today = datetime.now().date()
    start_date = st.date_input("Seleziona il giorno di inizio settimana", value=today)
    days_of_week = [start_date + timedelta(days=i) for i in range(7)]
    
    # Mappa abbreviata giorni in italiano
    giorni_settimana_it = {
        "Mon": "Lun",
        "Tue": "Mar",
        "Wed": "Mer",
        "Thu": "Gio",
        "Fri": "Ven",
        "Sat": "Sab",
        "Sun": "Dom"
    }
    days_labels = [giorni_settimana_it[day.strftime("%a")] + day.strftime(" %d/%m") for day in days_of_week]
    
    # 3. Prepara lista utenti
    users_list = supabase.table("profiles").select("*").execute().data
    utenti = sorted([f"{u['nome']} {u['cognome']}" for u in users_list])
    
    # 4. Costruisci matrice ferie (utente x giorno)
    ferie_matrix = []
    for utente in utenti:
        row = []
        ferie_utente = ferie_df[ferie_df["NOME"] == utente]
        ferie_utente = ferie_utente.copy()
        for idx, r in ferie_utente.iterrows():
            try:
                ferie_utente.at[idx, "DATA INIZIO"] = datetime.strptime(r["DATA INIZIO"], "%d/%m/%Y").date()
                ferie_utente.at[idx, "DATA FINE"] = datetime.strptime(r["DATA FINE"], "%d/%m/%Y").date()
            except Exception:
                ferie_utente.at[idx, "DATA INIZIO"] = None
                ferie_utente.at[idx, "DATA FINE"] = None
    
        for giorno in days_of_week:
            in_ferie = False
            motivo = ""
            for _, r in ferie_utente.iterrows():
                if r["DATA INIZIO"] and r["DATA FINE"] and r["DATA INIZIO"] <= giorno <= r["DATA FINE"]:
                    in_ferie = True
                    motivo = r.get("MOTIVO", "")
            if in_ferie:
                if motivo == "Malattia":
                    row.append("🇨🇭" + (f" {motivo}" if motivo else ""))
                else:
                    row.append("🌴" + (f" {motivo}" if motivo else ""))
            else:
                row.append("")
        ferie_matrix.append(row)
    
    # 5. Visualizza tabella con celle evidenziate e centrata
    ferie_report_df = pd.DataFrame(ferie_matrix, columns=days_labels, index=utenti)
    
    def evidenzia_ferie(val):
        if isinstance(val, str) and val.startswith("🌴"):
            return 'background-color: #E6F7DD; text-align: center;'
        elif isinstance(val, str) and val.startswith("🇨🇭"):
            return 'background-color: #FFA1A1; text-align: center;'
        return 'text-align: center;'
    
    ferie_report_df_styled = (
        ferie_report_df
        .style
        .applymap(evidenzia_ferie)
        .set_table_styles([
            {"selector": "th", "props": [("font-weight", "normal"), ("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
        .set_properties(**{"text-align": "center"})
    )
    
    st.markdown("""
        <style>
            .streamlit-expander, .block-container, table {
                margin-left: auto !important;
                margin-right: auto !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(ferie_report_df_styled.to_html(escape=False), unsafe_allow_html=True)
