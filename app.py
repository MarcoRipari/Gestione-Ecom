# app.py
import streamlit as st
from supabase import create_client
import jwt
from dataclasses import dataclass
import plotly.express as px
from streamlit_elements import elements, mui
from streamlit_chat import message

# -------------------------
# Config Supabase
# -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
REDIRECT_URL = "https://gestione-ecom.streamlit.app/"

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# -------------------------
# User dataclass
# -------------------------
@dataclass
class User:
    id: str
    email: str
    name: str
    role: str

# -------------------------
# Google OAuth link
# -------------------------
def google_oauth_url():
    return f"{SUPABASE_URL}/auth/v1/authorize?provider=google&redirect_to={REDIRECT_URL}"

# -------------------------
# Carica utente dal token
# -------------------------
def load_user():
    if "user" in st.session_state:
        return
    
    query_params = st.query_params  # <-- nuovo API
    if "access_token" in query_params:
        token = query_params["access_token"][0]
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            email = payload.get("email")
            name = payload.get("name")
            
            profile_resp = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
            profile = profile_resp.data
            role = profile["role"] if profile else "viewer"
            
            st.session_state["user"] = User(id=user_id, email=email, name=name, role=role)
            st.session_state["access_token"] = token
        except Exception as e:
            st.error(f"Errore nel caricamento utente: {e}")

# -------------------------
# Controllo login
# -------------------------
load_user()

if "user" not in st.session_state:
    st.markdown(
        f"<a href='{google_oauth_url()}'><button style='padding:10px 20px;border-radius:8px;background:#4f46e5;color:white;border:none;'>Login con Google</button></a>",
        unsafe_allow_html=True
    )
    st.stop()

user = st.session_state["user"]

# -------------------------
# Logout
# -------------------------
if st.sidebar.button("Logout"):
    st.session_state.pop("user", None)
    st.experimental_rerun()

# -------------------------
# Topbar
# -------------------------
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;padding:1rem;background:#1f2937;color:white;border-bottom:2px solid #4f46e5;'>
    <div style='display:flex;align-items:center;gap:1rem;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3652/3652191.png' width='32'/>
        <h3 style='margin:0'>Pro Business App</h3>
    </div>
    <div>
        <input placeholder='Cerca...' style='padding:6px 12px;border-radius:8px;border:none;background:#374151;color:white;'/>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar con sotto-menu
# -------------------------
st.sidebar.write(f"Benvenuto, {user.name} ({user.role})")
main_menu = st.sidebar.selectbox("Seleziona pagina", ["Dashboard","Data","Editor","Chat","Settings","About"])
submenu = None
if main_menu == "Data":
    options = ["Overview"]
    if user.role in ["editor","admin"]:
        options += ["Analytics","Export"]
    submenu = st.sidebar.selectbox("Opzioni Data", options)

# -------------------------
# Funzioni pagine
# -------------------------
def require_role(roles):
    if user.role not in roles:
        st.error("Accesso negato!")
        st.stop()

def page_dashboard():
    require_role(["viewer","editor","admin"])
    st.subheader("Dashboard")
    with elements("metrics", key="dash_metrics"):
        mui.Card(style={"margin":"10px","padding":"10px","backgroundColor":"#1f2937","color":"white"}).write("Metriche principali")
    fig = px.bar(x=["A","B","C"], y=[10,20,30], title="Grafico esempio")
    st.plotly_chart(fig, use_container_width=True)

def page_data(submenu=None):
    require_role(["viewer","editor","admin"])
    st.subheader(f"Data - {submenu or 'Overview'}")
    st.write("Qui inserisci le tabelle dati, grafici, export…")

def page_editor():
    require_role(["editor","admin"])
    st.subheader("Editor")
    st.write("Area di modifica dati")

def page_chat():
    require_role(["viewer","editor","admin"])
    st.subheader("Chat")
    message("Ciao! Questa è una chat demo", is_user=False)

def page_settings():
    require_role(["admin"])
    st.subheader("Settings")
    st.write("Solo admin possono modificare impostazioni globali")

def page_about():
    st.subheader("About")
    st.write("App professionale multiutente con Google OAuth e Streamlit Cloud")

# -------------------------
# Mappa pagine
# -------------------------
PAGE_MAP = {
    "Dashboard": page_dashboard,
    "Data": page_data,
    "Editor": page_editor,
    "Chat": page_chat,
    "Settings": page_settings,
    "About": page_about
}

# -------------------------
# Mostra pagina selezionata
# -------------------------
if main_menu == "Data":
    PAGE_MAP[main_menu](submenu)
else:
    PAGE_MAP[main_menu]()
