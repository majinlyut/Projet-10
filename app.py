import os
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# --- Mistral API key ---
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-small"

if not api_key:
    st.error("❌ Clé API Mistral non trouvée.")
    st.stop()

client = MistralClient(api_key=api_key)

# --- Embeddings wrapper ---
class CustomMistralEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
    def embed_documents(self, texts):
        response = self.client.embeddings(model="mistral-embed", input=texts)
        return [res.embedding for res in response.data]
    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding_function = CustomMistralEmbeddings(client)

# --- Chargement index FAISS ---
try:
    vectorstore = FAISS.load_local(
        "faiss_langchain_index",
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    st.error(f"❌ Erreur chargement index FAISS : {e}")
    st.stop()

# --- Prompt système  ---
SYSTEM_PROMPT = """Tu es un assistant culturel pour Paris.

Ta mission est de recommander des événements aux utilisateurs en te basant uniquement sur le CONTEXTE fourni ci-dessous.

Lorsque tu cites un événement respecte le format suivant :
📌 **{{titre}}**  
📍 _Lieu : {{lieu}}_ 
🏠 _Adresse : {{adresse}}_ 
🗓️ _du {{date de début}} au {{date de fin}}_  
📝 {{courte description}}



- Sépare chaque événement par une ligne vide.
- Formate les dates comme ca: "du 1er janvier au 5 février 2025" ou si une seule date: : "le 1er janvier 2025"
- N’ajoute pas de lien externe.
- Si la question utilisateur n'est pas claire, demande des précisions.
- Si la question est hors sujet, indique que tu ne réponds que sur les événements ayant lieu à Paris.
- Si tu as répondu à la question de l'utilisateur, termine par "As-tu d'autres questions ?"


---

CONTEXTE :
{context_str}

---

QUESTION :
{question}

RÉPONSE DE L'ASSISTANT :
"""

# --- Historique conversationnel ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Salut 👋 Je peux te recommander des événements à Paris. Pose-moi ta question !"}
    ]

st.title("🎙️ Assistant Événements à Paris")

# --- Affichage historique des messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Formatage dynamique des documents ---
def format_doc(doc, score):
    title = doc.metadata.get("title", "Titre inconnu")
    lieu = doc.metadata.get("location_name", "Lieu inconnu")
    adresse = doc.metadata.get("location_address", "Adresse inconnue")
    debut = doc.metadata.get("firstdate_begin", "?")
    fin = doc.metadata.get("lastdate_end", "?")
    description = doc.page_content.strip()

    return (
        f"📌 **{title}**  \n"
        f"📍 _Lieu : {lieu}_  \n"
        f"🏠 _Adresse : {adresse}_  \n"
        f"🗓️ _du {debut} au {fin}_  \n"
        f"📝 {description}"
    )



# --- Entrée utilisateur ---
if user_input := st.chat_input("Quel type d'événement t'intéresse ?"):

    # Affiche le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Recherche vectorielle
    try:
        results = vectorstore.similarity_search_with_score(user_input, k=4)
    except Exception:
        st.error("❌ Erreur pendant la recherche vectorielle.")
        results = []

    if not results:
        context_str = "Aucune information pertinente trouvée."
    else:
        context_str = "\n\n---\n\n".join([format_doc(doc, score) for doc, score in results])

    # Génération du prompt complet
    final_prompt = SYSTEM_PROMPT.format(context_str=context_str, question=user_input)

    # Appel à l’API Mistral
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⌛")

        try:
            response = client.chat(
                model=model,
                messages=[ChatMessage(role="user", content=final_prompt)],
                temperature=0.2,
                top_p=0.9
            )
            assistant_reply = response.choices[0].message.content
        except Exception:
            assistant_reply = "Désolé, je n’ai pas pu traiter ta demande. Réessaie plus tard."

        placeholder.markdown(assistant_reply, unsafe_allow_html=False)

    # Ajout dans l'historique
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
