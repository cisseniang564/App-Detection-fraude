import streamlit as st

from ui import hero, inject_css, section, stats, steps

inject_css()

hero(
    "Machine Learning · Assurance",
    "Détection de fraude à l'assurance auto",
    "Un pipeline complet — exploration, preprocessing, modélisation et prédiction en "
    "direct — pour repérer les sinistres suspects à partir des caractéristiques "
    "déclarées. Reproduit fidèlement la méthodologie du notebook original, avec vos "
    "propres données.",
)

stats([
    ("34", "colonnes attendues dans le fichier source"),
    ("3", "modèles comparés : Logistique, Random Forest, XGBoost"),
    ("2", "scénarios : données brutes et ré-équilibrées"),
    ("100%", "exécuté sur vos données, rien n'est pré-chargé"),
])

section("Comment ça marche")
steps([
    "<b>Données</b> — importez votre fichier (.xlsx ou .csv) au format du dataset "
    "d'origine (34 colonnes : caractéristiques du sinistre, de l'assuré et du véhicule).",
    "<b>Modélisation</b> — les 3 modèles sont entraînés automatiquement, d'abord sur "
    "les données brutes (déséquilibrées), puis sur une version ré-équilibrée par "
    "sur-échantillonnage — exactement la démarche du notebook.",
    "<b>Prédiction</b> — remplissez les caractéristiques d'un sinistre pour obtenir "
    "une probabilité de fraude en direct, avec le meilleur modèle entraîné.",
])

section("Pourquoi le ré-équilibrage change tout")
st.html(
    '<div class="card"><p>Sur des données brutes, la fraude est rare (environ 6&nbsp;% '
    "des sinistres) : un modèle qui prédit toujours « pas de fraude » obtient déjà "
    "~94&nbsp;% de précision — sans détecter un seul cas réel. C'est pour cette raison "
    "que l'onglet Modélisation affiche aussi le <b>rappel</b> (recall) sur la classe "
    "frauduleuse, pas seulement l'accuracy globale : c'est la métrique qui compte "
    "vraiment ici.</p></div>"
)

st.page_link("views/donnees.py", label="Commencer par importer des données", icon=":material/arrow_forward:")
