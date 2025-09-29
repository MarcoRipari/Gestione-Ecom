import streamlit as st
import functions

def view():
    ferie = functions.gsheet.get_sheet(st.secrets["FERIE_GSHEET_ID"], "FERIE").get_all_values()
    in_ferie = []
    for row in ferie[1:]:
        st.write(row)
    
    """Disegna la homepage"""
    st.title("🏠 Benvenuto nella HomePage2")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📅 Attività del giorno")
        st.write("• Task 1")
        st.write("• Task 2")

    with col2:
        st.subheader("🗂️ Catalogo")
        st.metric("Articoli totali", 1234)
        st.metric("Foto mancanti", 87)

    with col3:
        st.subheader("🌴 Ferie oggi")
        ferie_oggi = ["Mario Rossi", "Anna Bianchi"]
        for nome in ferie_oggi:
            st.write(f"• {nome}")
