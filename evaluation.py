import os
import pandas as pd
import asyncio
import nest_asyncio
import traceback
from datasets import Dataset
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Permet d'exécuter du code asynchrone dans un script Jupyter ou standard
nest_asyncio.apply()

# --- Jeu de test à remplir manuellement ---
questions_test = [
    "Quels sont les concerts gratuits en plein air?",
    "Je cherche une expo photo",
    "Je cherche un événement pour enfants"
]

answers = [
'''voici deux événements gratuits en plein air à paris : fête de la musique aux jardins du palais-royal lieu : jardins du domaine national du palais-royal adresse : place colette, 75001 paris le 21 juin 2024 concerts gratuits, mise en lumière de groupes amateurs, déambulations festives dans les rues. au programme : un concert sous casque avec "la muse en circuit" et des fanfares de tous les styles musicaux. open air escape lieu : route de boulogne à passy adresse : route de boulogne à passy, 75016 paris le 21 juin 2024 garden party en open air avec dj set (house/disco/afro house/tech house/techno). entrée gratuite pour une soirée festive en plein air. as-tu d'autres questions ?
''',
'''📌 exposition photos 📍 Lieu : église orthodoxe saint-serge 🏠 Adresse : 93 rue de crimée 75019 paris 🗓️ du 21 au 22 septembre 2024 📝 Visite de l'exposition photos dans l'église Saint-Serge.

📌 visite de la cité refuge 📍 Lieu : cité de refuge 🏠 Adresse : 12 rue cantagrel 75013 paris 🗓️ du 20 au 21 septembre 2024 📝 Exposition photos du Samu Social organisée durant cette période dans le hall.

As-tu d’autres questions ?''',
'''Voici deux événements adaptés aux enfants à Paris :

📌 visite guidée « en famille avec les grandes grandes vacances » 📍 Lieu : musée de la libération de paris - musée du général leclerc - musée jean moulin 🏠 Adresse : 4 avenue du colonel henri rol-tanguy 75014 paris 🗓️ le 21 septembre 2024 📝 Pour les enfants de 8 à 12 ans et leurs parents.

📌 lectures contées bilingues en famille en langue des signes française (lsf) et en français 📍 Lieu : musée zadkine 🏠 Adresse : 100 bis rue d'assas 75006 paris 🗓️ le 21 septembre 2024 📝 En famille, enfants à partir de 4 ans, activité proposée dans le cadre du week-end en famille spécial JEP 2024, accès libre et gratuit, dans la limite des places disponibles.

As-tu d’autres questions ?'''
]

ground_truths = [
'''Voici deux événements gratuits en plein air à Paris :

📌 Fête de la musique aux jardins du Palais-Royal 📍 Lieu : Jardins du domaine national du Palais-Royal 🏠 Adresse : Place Colette, 75001 Paris 🗓️ le 21 juin 2024 📝Au programme : un concert sous casque avec "La Muse en Circuit" et des fanfares de tous les styles musicaux.

📌 Open Air Escape 📍 Lieu : Route de Boulogne à Passy 🏠 Adresse : Route de Boulogne à Passy, 75016 Paris 🗓️ le 21 juin 2024 📝 Garden party en open air avec DJ set (House/Disco/Afro House/Tech House/Techno)

As-tu d’autres questions ?''',
    '''📌 exposition photos 📍 Lieu : église orthodoxe saint-serge 🏠 Adresse : 93 rue de crimée 75019 paris 🗓️ du 21 au 22 septembre 2024 📝 Visite de l'exposition photos dans l'église Saint-Serge.

📌 visite de la cité refuge 📍 Lieu : cité de refuge 🏠 Adresse : 12 rue cantagrel 75013 paris 🗓️ du 20 au 21 septembre 2024 📝 Exposition photos du Samu Social.

As-tu d’autres questions ?''',
    '''Voici deux événements adaptés aux enfants à Paris :

📌 visite guidée « en famille avec les grandes grandes vacances » 📍 Lieu : musée de la libération de paris - musée du général leclerc - musée jean moulin 🏠 Adresse : 4 avenue du colonel henri rol-tanguy 75014 paris 🗓️ le 21 septembre 2024 📝 A quoi ressemblait la vie en france durant la seconde guerre mondiale ?

📌 lectures contées bilingues en famille en langue des signes française (lsf) et en français 📍 Lieu : musée zadkine 🏠 Adresse : 100 bis rue d'assas 75006 paris 🗓️ le 21 septembre 2024 📝 Les bibliothécaires de la médiathèque de la canopée proposent des lectures contées en français et en lsf (langue des signes française)

As-tu d’autres questions ?'''
]


placeholder_contexts = [
    [
  "fête de la musique aux jardins du palais-royal - jardins du domaine national du palais-royal - place colette, 75001 paris - du 2024-06-21t11:30:00+00:00 au 2024-06-21t19:30:00+00:00 - concerts gratuits, mise en lumière de groupes amateurs, déambulations festives dans les rues… ce ne sont pas les animations qui manquent ! et si chaque restaurant, bar ou institution culturelle de la capitale met la main à la pâte pour faire de cet événement une grande célébration collective, cette année encore, le ministère de la culture propose une programmation inédite au cœur des jardins du palais-royal.",
  "fête de la musique aux jardins du palais-royal - jardins du domaine national du palais-royal - place colette, 75001 paris - du 2024-06-21t11:30:00+00:00 au 2024-06-21t19:30:00+00:00 - au programme : un concert sous casque avec \"la muse en circuit\" et des fanfares de tous les styles musicaux. une journée de concerts gratuits dans les jardins du palais-royal de 13h30 jusqu’à 21h15 ce vendredi 21 juin, les jardins du palais-royal se transformeront en véritable temple de la musique ! en accès libre, cette journée de concerts sera ouverte à tous les mélomanes, petits et grands.",
  "concert de la maîtrise notre-dame de paris - parvis de la cathédrale notre-dame de paris - parvis notre-dame de paris - du 2024-09-20t18:00:00+00:00 au 2024-09-21t18:30:00+00:00 - chants traditionnels et extraits du dernier disque de la maîtrise seront au programme : venez partager ce moment musical en plein air !",    
  "open air escape - rte de boulogne à passy, 75016 paris, france - rte de boulogne à passy, 75016 paris, france - du 2024-06-21t16:00:00+00:00 au 2024-06-21t21:55:00+00:00 - - house/disco/afro house/tech house/techno entrée gratuite. que vous soyez là pour célébrer avec des amis, rencontrer de nouvelles personnes, ou simplement profiter d'une soirée en plein air dans une ambiance festive, notre garden party en open air avec dj set est l'événement incontournable pour cette édition 2024 de la fête de la musique . venez vivre une nuit mémorable où la nature et la musique se conjuguent pour créer des moments magiques ."
],
    [
  "exposition photos - église orthodoxe saint-serge - 93 rue de crimée 75019 paris - du 2024-09-21t13:00:00+00:00 au 2024-09-22t15:00:00+00:00 - visite de l'exposition photos dans l'église saint-serge. visite de l'exposition photos dans l'église saint-serge",
  "barto+barto. trait pour trait - école nationale supérieure d'architecture de paris-la villette - 144 rue de flandre 75019 paris - du 2024-10-18t08:00:00+00:00 au 2024-10-19t16:00:00+00:00 - exposition présentée dans la salle d'exposition du 3 au 24 octobre 2024, du lundi au vendredi de 10h à 19h, samedi de 10h à 18h. entrée libre.",
  "visite de la cité refuge - cité de refuge - 12 rue cantagrel 75013 paris - du 2024-09-20t08:30:00+00:00 au 2024-09-21t13:30:00+00:00 - exposition photos du samu social organisée durant cette période dans hall.",
  "visite guidée de l'exposition \"paris-cinés 1982-1992\" - bibliothèque andré malraux - 112, rue de rennes 75006 paris - du 2024-09-28t13:00:00+00:00 au 2024-09-28t14:30:00+00:00 - une sensibilisation à la conservation de ce patrimoine culturel que le monde nous envie. une visite commentée en présence du photographe aura lieu le 28 septembre à 15h. sur réservation uniquement. jean-françois chaput est un photographe indépendant né en 1955 à paris. il a été assistant-réalisateur et projectionniste. son projet sur les cinémas parisiens développé pendant dix ans, a pris une très grande place dans sa vie et dans son travail. il a été exposé à plusieurs reprises."
],    [
  "visite guidée « en famille avec les grandes grandes vacances » - musée de la libération de paris - musée du général leclerc - musée jean moulin - 4 avenue du colonel henri rol-tanguy 75014 paris - du 2024-09-21t13:30:00+00:00 au 2024-09-21t15:00:00+00:00 - pour les enfants de 8 à 12 ans et leurs parents.",
  "journées du patrimoine culturel et naturel à la mairie du 9e - mairie du 9e - 6 rue drouot paris - du 2024-09-21t08:00:00+00:00 au 2024-09-22t16:00:00+00:00 - quant aux enfants, un grand nombre d'animations et ateliers leurs seront proposés.",
  "atelier lecture pour enfants - jardin partagé des olympiades - 59 rue nationale 75013 paris - du 2024-06-30t14:30:00+00:00 au 2024-06-30t15:00:00+00:00 - lecture de récits pour enfants.",
  "lectures contées bilingues en famille en langue des signes française (lsf) et en français - musée zadkine - 100 bis rue d'assas 75006 paris - du 2024-09-21t12:30:00+00:00 au 2024-09-21t15:00:00+00:00 - en famille, enfants à partir de 4 ans activité proposée dans le cadre du week-end en famille spécial jep 2024 accès libre et gratuit, dans la limite des places disponibles"
]    ]

# --- Création du dataset d'évaluation
evaluation_data = {
    "question": questions_test,
    "answer": answers,
    "contexts": placeholder_contexts,
    "ground_truth": ground_truths
}
evaluation_dataset = Dataset.from_dict(evaluation_data)
print("✅ Dataset d'évaluation prêt.")

# --- Configuration et Exécution de l'Évaluation
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "VOTRE_CLE_API_MISTRAL_ICI")
if MISTRAL_API_KEY == "VOTRE_CLE_API_MISTRAL_ICI" or not MISTRAL_API_KEY:
    print("⚠️ Clé API Mistral non trouvée.")

try:
    # Initialisation du modèle et des embeddings
    mistral_llm = ChatMistralAI(
        mistral_api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=0.1
    )
    mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]

    print("📊 Lancement de l'évaluation RAGAS (cela peut prendre quelques minutes)...")
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics_to_evaluate,
        llm=mistral_llm,
        embeddings=mistral_embeddings
    )

    # Résultats complets
    results_df = results.to_pandas()
    print("\n--- Résultats détaillés ---")
    print(results_df)

    # Moyennes globales
    print("\n✅ Scores moyens sur le dataset :")
    print(results_df.mean(numeric_only=True))

except Exception as e:
    print("❌ ERREUR durant l’évaluation :")
    traceback.print_exc()
