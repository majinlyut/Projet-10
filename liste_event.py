"""
liste_event.py

Script de prétraitement des événements publics parisiens issus de la plateforme OpenAgenda.

Fonctionnalités :
- Récupération des données en CSV via l'API publique OpenAgenda
- Normalisation des champs texte (suppression HTML, minuscule, nettoyage)
- Filtrage automatique :
    * Ville = Paris
    * Date de début ≥ il y a 12 mois (filtré côté API)
- Nettoyage :
    * Suppression des colonnes et lignes vides
    * Suppression des doublons stricts
- Export CSV final : `evenements_paris.csv`
"""

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from io import StringIO

def normalize_text(text):
    """
    Nettoie un champ texte brut.

    - Supprime les balises HTML
    - Convertit le texte en minuscules
    - Conserve les URL intactes
    - Supprime les caractères parasites

    Args:
        text (str or None): Texte d'entrée (souvent issu d'un champ CSV)

    Returns:
        str or None: Texte nettoyé
    """
    if pd.isna(text):
        return text

    if re.match(r'^https?://', str(text).strip()):
        return text  # On ne touche pas aux liens

    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ", strip=True)
    return text.lower()

# -------- RÉCUPÉRATION DES DONNÉES --------

one_year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/exports/csv"

params = {
    'where': f"location_city = 'Paris' AND firstdate_begin >= '{one_year_ago}'",
    'lang': 'fr',
    'use_labels': 'true',
    'delimiter': ';'
}

response = requests.get(url, params=params)

# -------- TRAITEMENT DES DONNÉES --------

if response.status_code == 200:
    df = pd.read_csv(StringIO(response.text), sep=';')
    print(f"📥 {len(df)} événements chargés avant nettoyage.")

    # Normalisation des textes (HTML, capitales, espaces)
    for col in df.select_dtypes(include=['object', 'string']):
        df[col] = df[col].map(normalize_text)

    # Suppression des colonnes entièrement vides
    df.dropna(axis=1, how='all', inplace=True)

    # Suppression des lignes entièrement vides
    df.dropna(axis=0, how='all', inplace=True)

    # Suppression des doublons exacts
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    after_dedup = len(df)
    print(f"🗑️ {before_dedup - after_dedup} doublon(s) supprimé(s).")

    # Sauvegarde finale
    df.to_csv('evenements_paris.csv', sep=';', index=False)
    print(f"✅ Fichier nettoyé enregistré : evenements_paris.csv")

else:
    print(f"❌ Erreur API : {response.status_code}")
    print(response.text)
