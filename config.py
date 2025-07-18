# 🌍 Mappatura codici lingua → nome completo
LANG_NAMES = {
    "IT": "italiano",
    "EN": "inglese",
    "FR": "francese",
    "DE": "tedesco"
}

# 🌍 Mappatura nome lingua → codice (inverso)
LANG_LABELS = {v.capitalize(): k for k, v in LANG_NAMES.items()}

# 🔢 Colonne predefinite per il prompt (se presenti nel CSV)
DEFAULT_COLUMNS = [
    "skuarticolo",
    "Classification",
    "Matiere", "Sexe",
    "Saison", "Silouhette",
    "shoe_toecap_zalando",
    "shoe_detail_zalando",
    "heel_height_zalando",
    "heel_form_zalando",
    "sole_material_zalando",
    "shoe_fastener_zalando",
    "pattern_zalando",
    "upper_material_zalando",
    "futter_zalando",
    "Subtile2",
    "Concept",
    "Sp.feature"
]

# 🎨 Toni disponibili per la descrizione
AVAILABLE_TONES = [
    "professionale",
    "amichevole",
    "accattivante",
    "descrittivo",
    "tecnico",
    "ironico",
    "minimal",
    "user friendly",
    "SEO-friendly"
]

# 🔁 Lunghezze standard per descrizioni
DESCRIPTION_LENGTHS = [str(i) for i in range(10, 110, 10)]

# 💾 Colonne da escludere da selezione per prompt
EXCLUDED_COLUMNS = ["Description", "Description2"]

# 📦 Sheet predefiniti
LOG_SHEET = "logs"
STORICO_PREFIX = "STORICO_"
