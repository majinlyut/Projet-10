import os
import time
import pandas as pd
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
import spacy

load_dotenv()

# -------- MISTRAL CLIENT --------
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("❌ MISTRAL_API_KEY non trouvée")
client = Mistral(api_key=api_key)

# -------- NLP spaCy --------
nlp = spacy.load("fr_core_news_sm")

def chunk_text_nlp(text, max_chars=500):
    """Découpe sémantique en blocs de max_chars sans couper les phrases"""
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

# -------- WRAPPER MISTRAL POUR LANGCHAIN --------
class CustomMistralEmbeddings(Embeddings):
    def __init__(self, client, batch_size=50, sleep_time=1.5, max_retries=5):
        self.client = client
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.max_retries = max_retries

    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._embed_batch(batch)
            if embeddings:
                all_embeddings.extend(embeddings)
            time.sleep(self.sleep_time)
        return all_embeddings

    def _embed_batch(self, texts):
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(model="mistral-embed", inputs=texts)
                return [res.embedding for res in response.data]
            except Exception as e:
                delay = 2 ** attempt
                print(f"⚠️ Erreur API : {e} — tentative {attempt+1}/{self.max_retries} dans {delay}s")
                time.sleep(delay)
        print("❌ Échec après plusieurs tentatives.")
        return []

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# -------- CHARGEMENT DU CSV --------
df = pd.read_csv('evenements_paris.csv', sep=';')
print(f"📥 {len(df)} événements chargés depuis le CSV.")

def get_first_valid(row, keys):
    for key in keys:
        val = row.get(key)
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return ""

# -------- PRÉPARATION DES DOCUMENTS --------
documents = []

for _, row in df.iterrows():
    titre = get_first_valid(row, ["Titre"])
    description = get_first_valid(row, ["Description"])
    longue_description = get_first_valid(row, ["Description longue", "Détail des conditions"])
    identifiant = get_first_valid(row, ["Identifiant"])

    full_text = f"{titre}. {description}. {longue_description}".strip()

    if full_text and full_text != "...":
        metadata = {
            "id": identifiant,
            "firstdate_begin": get_first_valid(row, ["Première date - Début"]),
            "lastdate_end": get_first_valid(row, ["Dernière date - Fin"]),
            "location_name": get_first_valid(row, ["Nom du lieu"]),
            "location_address": get_first_valid(row, ["Adresse"]),
        }

        chunks = chunk_text_nlp(full_text, max_chars=500)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata=metadata))

print(f"📝 {len(documents)} chunks prêts pour l'embedding.")

# -------- VERIFICATION : NOMBRE D'ÉVÉNEMENTS INDEXÉS --------
indexed_event_ids = set(doc.metadata.get("id") for doc in documents if doc.metadata.get("id"))
csv_event_ids = set(str(row["Identifiant"]).strip() for _, row in df.iterrows() if pd.notna(row["Identifiant"]))

print(f"\n📋 Événements dans le CSV : {len(csv_event_ids)}")
print(f"📌 Événements indexés (au moins un chunk) : {len(indexed_event_ids)}")
print(f"✅ Taux de couverture : {len(indexed_event_ids) / len(csv_event_ids) * 100:.2f}%")

# -------- EMBEDDING + INDEXATION --------
embedding_function = CustomMistralEmbeddings(client)
vectorstore = FAISS.from_documents(documents, embedding_function)

# -------- SAUVEGARDE DE L'INDEX --------
vectorstore.save_local("faiss_langchain_index")
print("✅ Index FAISS LangChain sauvegardé dans 'faiss_langchain_index/'")

# -------- RECHERCHE DE TEST --------
print("\n🔎 Lancement de la recherche de test...")
query = "concert en plein air à Paris"
results = vectorstore.similarity_search(query, k=3)
print(f"📊 {len(results)} résultats trouvés")

for i, doc in enumerate(results, 1):
    lieu = doc.metadata.get("location_name", "Lieu inconnu")
    extrait = doc.page_content[:100].replace("\n", " ")
    print(f"{i}. 📍 {lieu} | 📝 {extrait}...")

print("\n✅ Script terminé")
