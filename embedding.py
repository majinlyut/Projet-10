"""
embedding.py

Script de création d'index vectoriel FAISS à partir d'événements parisiens.

Étapes :
- Chargement d'un CSV nettoyé (`evenements_paris.csv`)
- Nettoyage + segmentation des textes avec spaCy
- Encodage des chunks en embeddings via Mistral AI
- Sauvegarde de l'index FAISS (pour recherche vectorielle ultérieure)
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from mistralai.client import MistralClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import spacy

# -------- INITIALISATION --------

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ MISTRAL_API_KEY non trouvée dans l'environnement")

client = MistralClient(api_key=api_key)
nlp = spacy.load("fr_core_news_sm")


def chunk_text_nlp(text, max_chars=500):
    """
    Segmente un texte en plusieurs chunks sans couper les phrases.

    Utilise spaCy pour découper proprement les phrases tout en respectant
    une taille maximale (en caractères).

    Args:
        text (str): le texte à segmenter
        max_chars (int): taille max par chunk

    Returns:
        List[str]: liste de segments textuels
    """
    doc = nlp(text)
    chunks = []
    buffer = ""

    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence:
            continue
        if len(buffer) + len(sentence) + 1 <= max_chars:
            buffer += " " + sentence if buffer else sentence
        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = sentence
    if buffer:
        chunks.append(buffer.strip())
    return chunks


class CustomMistralEmbeddings(Embeddings):
    """
    Wrapper pour utiliser les embeddings Mistral avec LangChain.

    Attributs :
        client (Mistral) : instance client Mistral
        batch_size (int) : taille des lots d'embedding
        sleep_time (float) : délai entre les requêtes
        max_retries (int) : nombre maximal de tentatives en cas d'erreur API
    """

    def __init__(self, client, batch_size=200, sleep_time=0.5, max_retries=5):
        self.client = client
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.max_retries = max_retries

    def embed_documents(self, texts):
        """
        Embedding de plusieurs documents en batchs.

        Args:
            texts (List[str]): textes à encoder

        Returns:
            List[List[float]]: vecteurs d'embedding
        """
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch)
            if embeddings:
                all_embeddings.extend(embeddings)
            time.sleep(self.sleep_time)
        return all_embeddings

    def _embed_batch(self, texts):
        """
        Envoie un batch de textes à l'API Mistral avec gestion des erreurs.

        Args:
            texts (List[str]): batch à encoder

        Returns:
            List[List[float]]: vecteurs encodés ou [] si échec
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings(model="mistral-embed", input=texts)
                return [res.embedding for res in response.data]
            except Exception as e:
                delay = 2 ** attempt
                print(f"⚠️ Erreur API : {e} — tentative {attempt+1}/{self.max_retries} dans {delay}s")
                time.sleep(delay)
        print("❌ Échec après plusieurs tentatives.")
        return []

    def embed_query(self, text):
        """Embedding d'une requête unique."""
        return self.embed_documents([text])[0]


def get_first_valid(row, keys):
    """
    Récupère la première valeur non vide dans une liste de colonnes possibles.

    Args:
        row (pd.Series): ligne du DataFrame
        keys (List[str]): noms de colonnes à tester

    Returns:
        str: valeur trouvée ou chaîne vide
    """
    for key in keys:
        val = row.get(key)
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return ""


# -------- CHARGEMENT DU CSV --------

df = pd.read_csv('evenements_paris.csv', sep=';')
print(f"📥 {len(df)} événements chargés depuis le CSV.")

# -------- PRÉPARATION DES DOCUMENTS --------

documents = []
for _, row in df.iterrows():
    
    description = get_first_valid(row, ["Description"])
    longue_description = get_first_valid(row, ["Description longue", "Détail des conditions"])
    identifiant = get_first_valid(row, ["Identifiant"])

    full_text = f"{description}. {longue_description}".strip()

    if full_text and full_text != "...":
        metadata = {
            "id": identifiant,
            "title" : get_first_valid(row, ["Titre"]),
            "firstdate_begin": get_first_valid(row, ["Première date - Début"]),
            "lastdate_end": get_first_valid(row, ["Dernière date - Fin"]),
            "location_name": get_first_valid(row, ["Nom du lieu"]),
            "location_address": get_first_valid(row, ["Adresse"]),
        }

        chunks = chunk_text_nlp(full_text, max_chars=500)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))

print(f"📝 {len(documents)} chunks prêts pour l'embedding.")

# -------- STATISTIQUES DE COUVERTURE --------

indexed_event_ids = set(doc.metadata.get("id") for doc in documents if doc.metadata.get("id"))
csv_event_ids = set(str(row["Identifiant"]).strip() for _, row in df.iterrows() if pd.notna(row["Identifiant"]))

print(f"\n📋 Événements dans le CSV : {len(csv_event_ids)}")
print(f"📌 Événements indexés (au moins un chunk) : {len(indexed_event_ids)}")
print(f"✅ Taux de couverture : {len(indexed_event_ids) / len(csv_event_ids) * 100:.2f}%")

# -------- EMBEDDING + INDEXATION --------

embedding_function = CustomMistralEmbeddings(client)
vectorstore = FAISS.from_documents(documents, embedding_function)

# -------- SAUVEGARDE --------

vectorstore.save_local("faiss_langchain_index")
print("✅ Index FAISS LangChain sauvegardé dans 'faiss_langchain_index/'")

# -------- TEST RECHERCHE --------

#print("\n🔎 Lancement de la recherche de test...")
#query = "concert en plein air à Paris"
#results = vectorstore.similarity_search(query, k=3)
#print(f"📊 {len(results)} résultats trouvés")

#for i, doc in enumerate(results, 1):
#    lieu = doc.metadata.get("location_name", "Lieu inconnu")
#    extrait = doc.page_content[:100].replace("\n", " ")
#    print(f"{i}. 📍 {lieu} | 📝 {extrait}...")

print("\n✅ Script terminé")
