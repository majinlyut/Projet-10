import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from io import StringIO



# -------- RÉCUPÉRATION DU CSV --------
one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/csv"

params = {
    'where': f"location_city = \'Paris\' AND firstdate_begin >= \'{one_year_ago}\'",
    'lang': 'fr',
    'use_labels': 'true',
    'delimiter': ';'
}

response = requests.get(url, params=params)

# -------- FONCTION DE NORMALISATION --------
def normalize_text(text):
    if pd.isna(text):
        return text

    # Vérifie si c'est une URL
    if re.match(r'^https?://', str(text).strip()):
        return text  # Ne touche pas aux liens

    # Sinon, normalisation classique
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ", strip=True)
    return text.lower()

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text), sep=';')
    print(f"📥 {len(df)} événements chargés avant nettoyage.")

    # -------- NETTOYAGE -------- analyse qualitative?

    # Normalisation des textes
    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].map(normalize_text)

    # Suppression des colonnes vides
    df.dropna(axis=1, how='all', inplace=True)

    # Suppression des lignes vides
    df.dropna(axis=0, how='all', inplace=True)

    # Suppression des doublons stricts
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    after_dedup = len(df)
    print(f"🗑️ {before_dedup - after_dedup} doublon(s) supprimé(s).")

    # Sauvegarde du CSV nettoyé
    df.to_csv('evenements_paris.csv', sep=';', index=False)
    print(f"✅ Fichier nettoyé enregistré : evenements_paris.csv")

else:
    print(f"❌ Erreur API : {response.status_code}")
    print(response.text)
