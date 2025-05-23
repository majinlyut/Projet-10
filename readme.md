# 🎙️ Assistant Événements Paris

Ce projet est un RAG qui recommande des événements culturels à Paris en s'appuyant sur la recherche sémantique via FAISS et les embeddings de Mistral AI.

---

## 🚀 Fonctionnalités

- Récupération automatique des événements parisiens depuis OpenAgenda (Open Data)
- Nettoyage et filtrage (ville = Paris, date ≥ il y a 12 mois)
- Indexation des descriptions via embeddings Mistral + FAISS
- Interface conversationnelle pour recommander des événements via Streamlit
- Historique conversationnel et prompt personnalisé
- Tests de validation sur les données (ville, date, doublons)

---

## 🧱 Stack utilisée

- 🧠 [MistralAI](https://mistral.ai) (`mistralai`) pour les embeddings et le modèle de chat
- 🗂️ FAISS via `langchain-community`
- 💬 `streamlit` pour l’interface utilisateur
- 🐍 `pandas`, `bs4`, `requests`, `spacy` pour le traitement des données
- 🧪 `pytest` pour les tests unitaires

---

## 🛠️ Installation

1. Clone le repo :

```bash
git clone https://github.com/ton-user/ton-repo.git
cd ton-repo
```
2.Crée un environnement virtuel :
```bash
python -m venv venv
venv\Scripts\activate 
```
3.Installe les dépendances :
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```
4.Crée un fichier .env avec ta clé API :
```bash
MISTRAL_API_KEY=ta_clé_mistral
```
⚙️ Indexation des événements
Lance ces script pour récupérer les événements + test de validation de données :
```bash
python liste_event.py
pytest -v
```
Lance ce script indexer les événements dans une base vectorielle FAISS :
```bash
python embedding.py
```
Cela va générer un dossier faiss_langchain_index/.

💬 Lancer l’assistant
```bash
streamlit run app.py
```
Tu pourras discuter avec l’assistant pour recevoir des suggestions d’événements à Paris.

📁 Arborescence des fichiers

```bash
.
├── app.py                 # Interface utilisateur Streamlit
├── liste_event.py         # Récupération + nettoyage des données OpenAgenda
├── embedding.py           # Embedding des événements et génération de l’index FAISS
├── test_donnee.py         # Tests unitaires sur les données exportées
├── evenements_paris.csv   # Données nettoyées
├── faiss_langchain_index/ # Index vectoriel sauvegardé
├── .env                   # Clé API Mistral
└── requirements.txt       # Dépendances Python

```