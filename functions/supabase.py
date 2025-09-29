import streamlit as st
from supabase import create_client, Client

supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
service_role_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)
supabase_admin = create_client(supabase_url, service_role_key)
