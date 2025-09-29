import streamlit as st
from . import supabase as sp

def login(username: str, password: str) -> bool:
    try:
        # 1. Recupera il profilo dallo username
        res_profile = supabase.table("profiles").select("*").eq("username", username).single().execute()
        if not res_profile.data:
            st.error("❌ Username non trovato")
            return False

        user_id = res_profile.data["user_id"]

        # 2. Recupera l'utente auth per ottenere l'email (richiede service_role_key)
        res_user = supabase_admin.auth.admin.get_user_by_id(user_id)
        email = res_user.user.email

        if not email:
            st.error("❌ Nessuna email trovata per questo utente")
            return False

        # 3. Login usando email + password
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if res.user:
            # 4. Salva in session_state
            st.session_state.user = {
                "email": email,
                "username": res_profile.data.get("username", ""),
                "nome": res_profile.data.get("nome", ""),
                "cognome": res_profile.data.get("cognome", ""),
                "role": res_profile.data.get("role", "guest"),
            }
            return True
        else:
            st.error("❌ Credenziali errate")
            return False

    except Exception as e:
        st.error(f"Errore login: {e}")
        return False


      
def login_password(email: str, password: str) -> bool:
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if res.user is not None:
           
            # Recupera il profilo dell'utente usando user_id
            profile = supabase.table("profiles").select("*").eq("user_id", res.user.id).single().execute()
            
            if profile.data is None:
                st.error("❌ Profilo utente non trovato")
                return False
            
            # Salva tutto in session_state
            st.session_state.user = {
                "data": res.user,
                "email": res.user.email,
                "nome": profile.data["nome"],
                "cognome": profile.data["cognome"],
                "username": profile.data["username"],
                "role": profile.data["role"]
            }
            #st.session_state.user = res.user
            #st.session_state.username = profile.data.get("username", res.user.email)
            return True
        else:
            st.error("❌ Email o password errati")
            return False
    except Exception as e:
        st.error(f"Errore login: {e}")
        return False


def logout():
    if "user" in st.session_state:
        supabase.auth.sign_out()
        st.session_state.user = None
        #st.session_state.username = None
        st.rerun()

def register_user(email: str, password: str, **param) -> bool:
    try:
        # 1. Crea l'utente in Supabase Auth
        res = supabase_admin.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })

        if not res.user:
            st.error("❌ Errore nella creazione utente in Auth")
            return False

        user_id = res.user.id

        # 2. Inserisci il profilo nella tabella profiles
        profile = {
            "user_id": user_id,
            "nome": param.get("nome", None),
            "cognome": param.get("cognome", None),
            "username": param.get("username", None),
            "role": param.get("role", None)
        }

        supabase_admin.table("profiles").insert(profile).execute()

        st.success(f"✅ Utente {username} creato correttamente")
        return True

    except Exception as e:
        st.error(f"Errore registrazione: {e}")
        return False
