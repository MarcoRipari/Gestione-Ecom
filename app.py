import streamlit as st
from supabase import create_client, Client
from dataclasses import dataclass

# --------------------------
# Config Supabase (da segreti Streamlit Cloud)
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
# Login / Logout con Supabase Google
# --------------------------
def login_with_google():
    # Avvia flusso login Google
    supabase.auth.sign_in_with_provider('google')

def logout():
    supabase.auth.sign_out()
    st.session_state['user'] = None
    st.experimental_rerun()

# --------------------------
# Controllo login e ruoli
# --------------------------
def load_user():
    if 'user' not in st.session_state or st.session_state['user'] is None:
        # Controlla se utente loggato con Supabase
        user = supabase.auth.user()
        if user:
            # Prendi profilo dal DB
            profile = supabase.table("profiles").select("*").eq("id", user.id).single().execute()
            role = profile.data['role'] if profile.data else "viewer"
            st.session_state['user'] = User(
                id=user.id,
                email=user.email,
                full_name=profile.data['full_name'] if profile.data else user.email,
                role=role
            )
        else:
            st.session_state['user'] = None

def require_role(roles):
    user = st.session_state.get("user")
    if not user or user.role not in roles:
        st.error("Accesso negato per il tuo ruolo!")
        st.stop()

# --------------------------
# Topbar
# --------------------------
def topbar():
    st.markdown("""
    <div style='display:flex;justify-content:space-between;align-items:center;padding:0.8rem 1.5rem;background:#1f2937;color:white;'>
        <div style='display:flex;align-items:center;gap:0.6rem;font-weight:600;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3652/3652191.png' width='28' height='28'/>
            <span>Pro Business App</span>
        </div>
        <div>
            <input type='text' placeholder='Search…' style='padding:6px 10px;border-radius:8px;border:none;background:#374151;color:white;' />
        </div>
    </div>
    """, unsafe_allow_html=True)

# --------------------------
# Sidebar con sotto-menu basato su ruoli
# --------------------------
def sidebar_menu():
    st.sidebar.write("### Menu")
    user = st.session_state['user']
    
    main_menu = st.sidebar.selectbox("Seleziona pagina", ["Dashboard", "Data", "Editor", "Chat", "Settings", "About"])
    submenu = None
    
    # Esempio sotto-menu basato sul ruolo
    if main_menu == "Data":
        options = ["Overview"]
        if user.role in ["editor","admin"]:
            options += ["Analytics","Export"]
        submenu = st.sidebar.selectbox("Opzioni Data", options)
    
    return main_menu, submenu

# --------------------------
# Pagine
# --------------------------
def page_dashboard():
    require_role(["viewer","editor","admin"])
    st.subheader("Dashboard")
    st.write("Contenuti visibili a tutti gli utenti loggati.")

def page_data(submenu=None):
    require_role(["viewer","editor","admin"])
    st.subheader(f"Data - {submenu or 'Overview'}")
    st.write("Contenuti della sezione Data.")

def page_editor():
    require_role(["editor","admin"])
    st.subheader("Editor")
    st.write("Solo editor/admin possono modificare qui.")

def page_chat():
    require_role(["viewer","editor","admin"])
    st.subheader("Chat")

def page_settings():
    require_role(["admin"])
    st.subheader("Settings")

def page_about():
    st.subheader("About")
    st.write("Pro Business App con Supabase Auth, login Google e multiutente.")

PAGE_MAP = {
    "Dashboard": page_dashboard,
    "Data": page_data,
    "Editor": page_editor,
    "Chat": page_chat,
    "Settings": page_settings,
    "About": page_about
}

# --------------------------
# Main
# --------------------------
def main():
    topbar()
    
    # Login / Load user
    load_user()
    user = st.session_state.get("user")
    
    if not user:
        st.title("Login con Google")
        if st.button("Login con Google"):
            login_with_google()
        st.stop()
    else:
        st.sidebar.write(f"Logged in as: {user.full_name} ({user.role})")
        if st.sidebar.button("Logout"):
            logout()
    
    # Sidebar e sotto-menu
    page, submenu = sidebar_menu()
    
    # Mostra pagina
    if page == "Data":
        PAGE_MAP[page](submenu)
    else:
        PAGE_MAP[page]()

if __name__ == "__main__":
    main()
