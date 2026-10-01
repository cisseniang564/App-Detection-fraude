# Détection de fraude à l'assurance auto — Streamlit

Application interactive reproduisant fidèlement le pipeline du notebook
`ProjetML_Predict_Claims_copy.ipynb` : exploration, preprocessing, entraînement
de 3 modèles (Régression logistique, Random Forest, XGBoost) sur données
brutes puis ré-équilibrées, et prédiction en direct sur un nouveau sinistre.

**Mode self-service** : l'app n'embarque aucun modèle pré-entraîné, mais
embarque désormais le **vrai jeu de données du projet** (`data/Dataset.xlsx`,
11 565 sinistres) — le bouton "Utiliser le jeu de données du projet" charge
ce fichier et entraîne les modèles en direct dessus. Tu peux aussi importer
ton propre fichier à la place (même schéma, 34 colonnes). Si `Dataset.xlsx`
venait à être retiré du dossier `data/`, l'app bascule automatiquement sur un
jeu synthétique de secours.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Déployer (Streamlit Community Cloud)

1. Pousser ce dossier sur GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → choisir le
   dépôt et `app.py`.

## Structure

```
fraud_app/
├── app.py               # Point d'entree, navigation
├── ui.py                # Composants HTML a styles inline (voir note ci-dessous)
├── pipeline.py           # Preprocessing + modelisation, pur et testable
├── views/
│   ├── accueil.py
│   ├── donnees.py         # Import + EDA
│   ├── modelisation.py    # Entrainement + comparaison des 3 modeles
│   └── prediction.py      # Formulaire de prediction en direct
├── requirements.txt
└── .streamlit/config.toml # Theme natif (couleurs, toolbar minimal)
```

## Note technique importante : pourquoi des styles inline ?

Une première version utilisait une feuille de style globale injectée via
`st.html("<style>...</style>")`. Ce mécanisme s'est avéré ne pas se monter
de façon fiable au premier chargement dans cet environnement de build
(plusieurs heures d'investigation — bug non élucidé avec certitude, non
reproduit avec un CSS minimal mais systématique avec le CSS complet combiné
au reste du contenu de page). Plutôt que de livrer quelque chose de fragile,
chaque composant (`ui.py`) applique désormais son style directement en
attribut `style="..."` sur chaque élément HTML, vérifié fiable à 100% sur de
nombreux essais. Le thème global (fond, couleur primaire, texte) passe par
`.streamlit/config.toml`, un mécanisme natif Streamlit qui ne dépend
d'aucune injection HTML côté client.

**Si tu ajoutes un nouveau composant visuel** : suis le même principe
(styles inline dans la fonction, pas de nouvelle feuille de style globale)
pour ne pas réintroduire le problème.

## Tests effectués avant livraison

- Tests unitaires du pipeline (`pipeline.py`) contre un jeu de données
  synthétique au schéma identique : preprocessing, entraînement brut et
  ré-équilibré, prédiction, robustesse aux modalités inconnues et à l'ordre
  des colonnes.
- Parcours complet (`streamlit.testing.v1.AppTest`) : Accueil → Données
  (import démo) → Modélisation (entraînement réel) → Prédiction (soumission),
  sans exception.
- Vérification visuelle réelle (Playwright/Chromium, desktop et mobile) sur
  les 4 pages, avec détection automatique de texte parasite, débordement
  horizontal et exceptions affichées à l'écran — répétée sur plusieurs
  chargements frais pour écarter tout problème de fiabilité au premier
  rendu.

## Résultats obtenus sur le vrai dataset (11 565 sinistres)

Vérifiés en conditions réelles avant livraison (split 70/30 stratifié, test
set non ré-équilibré — contrairement à un upsampling avant split, qui biaise
l'évaluation) :

| | Accuracy | Rappel fraude | Précision fraude | AUC |
|---|---|---|---|---|
| **Données brutes** — XGBoost | 94,2 % | 1,9 % | 100 % | 0,830 |
| **Données ré-équilibrées** — Régression logistique | 66,6 % | 88,8 % | 13,9 % | 0,804 |
| **Données ré-équilibrées** — Random Forest | 73,8 % | 73,3 % | 15,0 % | 0,829 |
| **Données ré-équilibrées** — XGBoost | 85,3 % | 40,8 % | 17,8 % | 0,817 |

Deux cas limites réels (absents du jeu synthétique de test) ont été corrigés
pendant la validation : des valeurs manquantes au-delà d'Age/DriverRating
(filet de sécurité généralisé à toute colonne), et une colonne
(`PastNumberOfClaims`) mélangeant entiers et chaînes de caractères pour une
même modalité — normalisée en texte à l'import.
