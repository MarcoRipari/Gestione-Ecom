import streamlit.components.v1 as components
import os

_RELEASE = True

if _RELEASE:
    _component_func = components.declare_component(
        "google_oauth_popup",
        path=os.path.join(os.path.dirname(__file__), "frontend/build"),
    )
else:
    _component_func = components.declare_component(
        "google_oauth_popup",
        url="http://localhost:3001",
    )

def google_login_button(oauth_url: str):
    """
    Mostra il pulsante Google OAuth popup.
    Restituisce l'access_token una volta che l'utente ha effettuato il login.
    """
    access_token = _component_func(url=oauth_url)
    return access_token
