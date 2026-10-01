"""App de detection de fraude a l'assurance auto - point d'entree."""
import streamlit as st

st.set_page_config(
    page_title="Détection de fraude — sinistres auto",
    page_icon="🚗",
    layout="wide",
)
# inject_css() n'est PAS appele ici : appele avant pg.run(), son <style> se
# retrouve hors du conteneur que st.navigation gere et n'est pas monte de
# facon fiable au premier chargement. Chaque page l'appelle elle-meme en
# premiere ligne (voir views/*.py) : fiable, verifie par test.

accueil = st.Page("views/accueil.py", title="Accueil", default=True)
donnees = st.Page("views/donnees.py", title="Données")
modelisation = st.Page("views/modelisation.py", title="Modélisation")
prediction = st.Page("views/prediction.py", title="Prédiction")

pg = st.navigation([accueil, donnees, modelisation, prediction], position="top")
pg.run()
