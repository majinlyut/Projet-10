import pandas as pd
from datetime import datetime, timedelta
import pytest

CSV_PATH = "evenements_paris.csv"

@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV_PATH, sep=';')

def test_ville_paris(df):
    assert "ville" in df.columns or any("ville" in col.lower() for col in df.columns), "Colonne contenant 'ville' introuvable"
    ville_col = [col for col in df.columns if "ville" in col.lower()][0]
    valeurs_uniques = df[ville_col].dropna().str.strip().str.lower().unique()
    assert all(v == "paris" for v in valeurs_uniques), f"Certaines villes ne sont pas Paris : {valeurs_uniques}"

def test_dates_superieures_a_moins_un_an(df):
    from datetime import datetime, timedelta, timezone

    date_col = [col for col in df.columns if "date" in col.lower() and "début" in col.lower()]
    assert date_col, "Colonne de date de début introuvable"

    df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')

    limite = datetime.now(timezone.utc) - timedelta(days=365)

    invalides = df[df[date_col[0]] < limite]

    assert invalides.empty, (
        f"{len(invalides)} événement(s) ont une date de début antérieure à il y a 1 an."
    )