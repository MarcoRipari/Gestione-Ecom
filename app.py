import streamlit as st
from supabase import create_client, Client
import jwt
from dataclasses import dataclass

# --------------------------
# Config Supabase
# --------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --------------------------
# User dataclass
# --------------------------
@dataclass
class User:
    id: str
    email: str
    full_name: str
    role: str

# --------------------------
# Login / Logout
# --------------------------
def login(email: str, password: str):
    auth = supabase.auth
    try:
        user = auth.sign_in_with_email(email, password)
        if user.user:
            # Ottieni profilo dal DB
            profile = supabase.table("profiles").select("*").eq("id", user.user.id).single().execute()
            role = profile.data['role'] if profile.data else "viewer"
            st.session_state['user'] = User(id=user.user.id, email=email, full_name=profile.data['full_name'], role=role)
            st.success(f"Benvenuto {st.session_state['user'].full_name}")
    except Exception as e:
        st.error(f"Errore login: {e}")

def logout():
    supabase.auth.sign_out()
    st.session_state['user'] = None
    st.experimental_rerun()

# --------------------------
# UI Login
# --------------------------
def login_ui():
    if "user" not in st.session_state or st.session_state["user"] is None:
        st.title("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            login(email, password)
        st.stop()
    else:
        st.sidebar.button("Logout", on_click=logout)

# --------------------------
# Accesso alle pagine
# --------------------------
def require_role(roles):
    user = st.session_state.get("user")
    if not user or user.role not in roles:
        st.error("Accesso negato per il tuo ruolo!")
        st.stop()

# --------------------------
# Esempio di pagina protetta
# --------------------------
def page_dashboard():
    require_role(["viewer","editor","admin"])
    st.subheader("Dashboard")
    st.write("Contenuti visibili a tutti gli utenti loggati")

def page_editor():
    require_role(["editor","admin"])
    st.subheader("Editor")
    st.write("Solo editor/admin possono modificare qui")

# --------------------------
# Main
# --------------------------
def main():
    login_ui()
    pages = {"Dashboard": page_dashboard, "Editor": page_editor}
    page = st.sidebar.selectbox("Seleziona pagina", list(pages.keys()))
    pages[page]()

if __name__ == "__main__":
    main()
