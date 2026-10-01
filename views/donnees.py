import plotly.graph_objects as go
import streamlit as st

from pipeline import (
    SchemaError, generate_demo_data, load_bundled_dataset, load_dataframe,
    preprocess, validate_schema,
)
from ui import GOLD, NAVY_2, WINE, hero, inject_css, reading, section, stats

inject_css()


# Mise en cache : sans elle, Streamlit relit le fichier Excel (~2-3s) et
# refait tout le preprocessing a CHAQUE interaction sur la page (changement
# d'onglet, clic sur un widget...), pas seulement au premier chargement --
# c'etait la cause principale de lenteur ressentie.
@st.cache_data(show_spinner="Lecture du jeu de données du projet…")
def _cached_bundled_dataset():
    return load_bundled_dataset()


@st.cache_data(show_spinner="Lecture du fichier importé…")
def _cached_load_dataframe(fichier):
    return load_dataframe(fichier)


@st.cache_data(show_spinner="Préparation des données…")
def _cached_preprocess(df):
    return preprocess(df)

hero(
    "Données",
    "Importer un jeu de données",
    "Fichier .xlsx ou .csv au format du dataset d'origine (34 colonnes). Sans fichier "
    "personnel à disposition, utilisez le jeu de données du projet (11 565 sinistres réels) "
    "pour explorer l'application.",
)

_bundled = _cached_bundled_dataset()

col_u, col_d = st.columns([3, 1], vertical_alignment="bottom")
with col_u:
    fichier = st.file_uploader("Fichier de sinistres (.xlsx, .csv)", type=["xlsx", "xls", "csv"])
with col_d:
    label_demo = "Utiliser le jeu de données du projet" if _bundled is not None else "Utiliser un jeu de démonstration"
    demo = st.button(label_demo, width="stretch")

if demo:
    if _bundled is not None:
        st.session_state["df"] = _bundled
        st.session_state["df_name"] = "Dataset.xlsx (jeu de données du projet, 11 565 sinistres)"
    else:
        st.session_state["df"] = generate_demo_data()
        st.session_state["df_name"] = "Jeu de démonstration (synthétique)"
    st.session_state.pop("results_brut", None)
    st.session_state.pop("results_balanced", None)

if fichier is not None:
    try:
        df_loaded = _cached_load_dataframe(fichier)
        st.session_state["df"] = df_loaded
        st.session_state["df_name"] = fichier.name
        st.session_state.pop("results_brut", None)
        st.session_state.pop("results_balanced", None)
    except Exception as exc:
        st.error(f"Impossible de lire le fichier : {exc}")

df = st.session_state.get("df")

if df is None:
    st.info("Importez un fichier ou utilisez le jeu de démonstration pour continuer.")
    st.stop()

st.caption(f"Source actuelle : **{st.session_state.get('df_name', 'fichier importé')}**")

missing = validate_schema(df)
if missing:
    st.error(
        "Ce fichier ne correspond pas au schéma attendu — colonnes manquantes : "
        + ", ".join(f"`{c}`" for c in missing)
    )
    st.stop()

try:
    X, y, warnings = _cached_preprocess(df)
    st.session_state["X"], st.session_state["y"] = X, y
except SchemaError as exc:
    st.error(str(exc))
    st.stop()

stats([
    (f"{df.shape[0]:,}".replace(",", "\u202f"), "sinistres"),
    (f"{df.shape[1]}", "colonnes"),
    (f"{y.mean() * 100:.1f}".replace(".", ",") + "\u202f%", "taux de fraude déclaré"),
    (f"{int(df.isna().sum().sum())}", "valeurs manquantes (avant imputation)"),
])

if warnings:
    st.html("<div>" + "".join(f'<p style="font-size:13.5px;color:#9C6B1F;">⚠ {w}</p>' for w in warnings) + "</div>")

section("Aperçu des données")
st.dataframe(df.head(20), width="stretch", height=280)

section("Déséquilibre de la classe cible")
counts = y.value_counts().sort_index()
fig = go.Figure(go.Bar(
    x=["Non frauduleux", "Frauduleux"], y=[counts.get(0, 0), counts.get(1, 0)],
    marker_color=[NAVY_2, WINE], text=[counts.get(0, 0), counts.get(1, 0)], textposition="outside",
))
fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   font=dict(family="Inter, sans-serif", size=13))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
reading(
    f"<b>Lecture :</b> {counts.get(1, 0)} sinistres frauduleux sur {len(y)} "
    f"({y.mean() * 100:.1f} %). C'est ce déséquilibre qui rend l'accuracy brute "
    "trompeuse — voir l'onglet Modélisation."
)

section("Taux de fraude par variable clé")
col1, col2 = st.columns(2)
for col, target_col in zip([col1, col2], ["Fault", "PoliceReportFiled"]):
    with col:
        tmp = df.groupby(target_col)["FraudFound_P"].mean().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=tmp.index.astype(str), y=tmp.values * 100, marker_color=GOLD))
        fig.update_layout(
            title=dict(text=f"Taux de fraude par « {target_col} »", font=dict(size=13)),
            height=260, margin=dict(l=10, r=10, t=36, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="% de fraude", font=dict(family="Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.page_link("views/modelisation.py", label="Passer à la modélisation", icon=":material/arrow_forward:")
