import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import train_all
from ui import GOLD, NAVY_2, hero, inject_css, reading, section

inject_css()

hero(
    "Modélisation",
    "Entraîner et comparer les modèles",
    "Régression logistique, Random Forest et XGBoost, entraînés sur les données "
    "brutes puis sur une version ré-équilibrée par sur-échantillonnage — la démarche "
    "du notebook, reproduite à l'identique.",
)

if "X" not in st.session_state:
    st.info("Importez d'abord des données dans l'onglet **Données**.")
    st.page_link("views/donnees.py", label="Aller à l'import des données", icon=":material/arrow_back:")
    st.stop()

X, y = st.session_state["X"], st.session_state["y"]

if st.button("🚀 Entraîner les 3 modèles (brut + ré-équilibré)", type="primary"):
    with st.spinner("Entraînement en cours — quelques secondes à quelques minutes selon la taille du fichier…"):
        st.session_state["results_brut"] = train_all(X, y, balanced=False)
        st.session_state["results_balanced"] = train_all(X, y, balanced=True)

if "results_brut" not in st.session_state:
    st.stop()

res_brut = st.session_state["results_brut"]
res_bal = st.session_state["results_balanced"]
MODELS = ["Régression logistique", "Random Forest", "XGBoost"]


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        m = results[name]["metrics_test"]
        fraude = m["report"].get("Frauduleux", {})
        rows.append({
            "Modèle": name,
            "Accuracy": m["accuracy"],
            "Rappel (fraude)": fraude.get("recall", 0.0),
            "Précision (fraude)": fraude.get("precision", 0.0),
            "F1 (fraude)": fraude.get("f1-score", 0.0),
            "AUC": m.get("auc", float("nan")),
        })
    return pd.DataFrame(rows).set_index("Modèle")


def show_confusion(results: dict, name: str) -> None:
    cm = results[name]["metrics_test"]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["Prédit : Non frauduleux", "Prédit : Frauduleux"],
        y=["Réel : Non frauduleux", "Réel : Frauduleux"],
        text=cm, texttemplate="%{text}", colorscale=[[0, "#EFEBE1"], [1, NAVY_2]],
        showscale=False,
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=12))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


tab_brut, tab_bal = st.tabs(["Données brutes (déséquilibrées)", "Données ré-équilibrées (upsampling)"])

with tab_brut:
    section("Comparaison des 3 modèles")
    tbl = comparison_table(res_brut)
    st.dataframe(tbl.style.format({
        "Accuracy": "{:.1%}", "Rappel (fraude)": "{:.1%}", "Précision (fraude)": "{:.1%}",
        "F1 (fraude)": "{:.1%}", "AUC": "{:.3f}",
    }), width="stretch")
    reading(
        "<b>Lecture :</b> l'accuracy est élevée (~94&nbsp;%) pour les trois modèles, mais regardez "
        "le <b>rappel sur la fraude</b> : souvent proche de 0. Les modèles se contentent de "
        "prédire la classe majoritaire — exactement le piège du déséquilibre de classe que "
        "le ré-équilibrage va corriger."
    )
    section("Matrice de confusion")
    choix_brut = st.selectbox("Modèle", MODELS, key="sel_brut")
    show_confusion(res_brut, choix_brut)

with tab_bal:
    section("Comparaison des 3 modèles")
    tbl = comparison_table(res_bal)
    st.dataframe(tbl.style.format({
        "Accuracy": "{:.1%}", "Rappel (fraude)": "{:.1%}", "Précision (fraude)": "{:.1%}",
        "F1 (fraude)": "{:.1%}", "AUC": "{:.3f}",
    }), width="stretch")
    best = tbl["F1 (fraude)"].idxmax()
    reading(
        f"<b>Lecture :</b> après ré-équilibrage, le rappel sur la fraude progresse nettement — "
        f"c'est le modèle à privilégier pour un usage opérationnel. Meilleur F1 (fraude) ici : "
        f"<b>{best}</b>. C'est ce modèle qui est utilisé par défaut dans l'onglet Prédiction."
    )
    section("Matrice de confusion")
    choix_bal = st.selectbox("Modèle", MODELS, key="sel_bal", index=MODELS.index(best))
    show_confusion(res_bal, choix_bal)

st.page_link("views/prediction.py", label="Passer à la prédiction en direct", icon=":material/arrow_forward:")
