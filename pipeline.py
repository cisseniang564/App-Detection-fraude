"""Pipeline de detection de fraude a l'assurance auto.

Reproduit fidelement le preprocessing et la modelisation du notebook original
(ProjetML_Predict_Claims_copy.ipynb) : encodage ordinal pour les variables a
modalites ordonnees, one-hot pour les variables nominales, puis entrainement
de 3 modeles (Regression Logistique, Random Forest, XGBoost) sur les donnees
brutes ET sur les donnees ré-equilibrees par sur-echantillonnage.

Toutes les fonctions sont pures (entree explicite, sortie explicite, aucun
effet de bord Streamlit) : testables independamment de l'interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from xgboost import XGBClassifier

# =============================================================================
# 1. SCHEMA DE REFERENCE (colonnes attendues et leurs modalites)
# =============================================================================

TARGET = "FraudFound_P"
ID_COL = "PolicyNumber"

# Colonnes numeriques brutes (inchangees par le preprocessing)
NUMERIC_COLS = [
    "WeekOfMonth", "WeekOfMonthClaimed", "Age", "RepNumber",
    "Deductible", "DriverRating", "Year", "ClaimSize",
]

# Variables categorielles ORDONNEES -> encodage ordinal (1 seule colonne,
# valeurs entieres respectant l'ordre naturel de la modalite)
ORDINAL_MAPS = {
    "Month": {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
              "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11},
    "MonthClaimed": {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "Jun": 5,
                      "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11},
    "DayOfWeek": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6},
    "DayOfWeekClaimed": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                          "Friday": 4, "Saturday": 5, "Sunday": 6},
    "VehiclePrice": {"less than 20000": 1, "20000 to 29000": 2, "30000 to 39000": 3,
                      "40000 to 59000": 4, "60000 to 69000": 5, "more than 69000": 6},
    "AgeOfVehicle": {"new": 1, "2 years": 2, "3 years": 3, "4 years": 4,
                      "5 years": 5, "6 years": 6, "7 years": 7, "more than 7": 8},
    "AgeOfPolicyHolder": {"16 to 17": 1, "18 to 20": 2, "21 to 25": 3, "26 to 30": 4,
                           "31 to 35": 5, "36 to 40": 6, "41 to 50": 7, "51 to 65": 8,
                           "over 65": 9},
    "Days_Policy_Accident": {"none": 0, "1 to 7": 1, "8 to 15": 2, "15 to 30": 3,
                              "more than 30": 4},
    "Days_Policy_Claim": {"none": 0, "8 to 15": 1, "15 to 30": 2, "more than 30": 3},
}

# Variables categorielles NOMINALES -> one-hot (pd.get_dummies, drop_first=True)
ONEHOT_COLS = [
    "Make", "AccidentArea", "Sex", "MaritalStatus", "Fault", "PolicyType",
    "VehicleCategory", "PastNumberOfClaims", "PoliceReportFiled",
    "WitnessPresent", "AgentType", "NumberOfSuppliments",
    "AddressChange_Claim", "NumberOfCars", "BasePolicy",
]

ALL_RAW_COLS = (
    NUMERIC_COLS + list(ORDINAL_MAPS.keys()) + ONEHOT_COLS + [TARGET, ID_COL]
)


class SchemaError(ValueError):
    """Le fichier importe ne correspond pas au schema attendu."""


# =============================================================================
# 1bis. JEU DE DONNEES DE DEMONSTRATION (meme schema exact, pour essayer l'app
#       sans posseder le fichier source)
# =============================================================================

DATASET_PATH = "data/Dataset.xlsx"


def _normalize_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force les colonnes categorielles (ordinales + one-hot) en texte pur.

    Un fichier Excel reel peut stocker une modalite comme nombre plutot que
    texte (ex. la modalite "1" de PastNumberOfClaims lue comme l'entier 1
    au lieu de la chaine "1") des que la colonne melange plusieurs types.
    Cela casse `sorted()` (comparaison int/str impossible), la serialisation
    Arrow de st.dataframe, et le mapping ordinal. On uniformise en texte,
    en preservant les valeurs reellement manquantes (NaN reste NaN, pas la
    chaine "nan").
    """
    df = df.copy()
    cols = list(ORDINAL_MAPS.keys()) + ONEHOT_COLS
    for col in cols:
        if col in df.columns:
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
    return df


def load_bundled_dataset() -> pd.DataFrame | None:
    """Charge le vrai jeu de donnees du projet (data/Dataset.xlsx) s'il est
    present dans le depot. Renvoie None si absent (repli sur les donnees
    synthetiques via generate_demo_data), pour que l'app reste utilisable
    meme sans ce fichier (ex. avant que l'utilisateur ne l'ajoute lui-meme)."""
    from pathlib import Path

    path = Path(__file__).resolve().parent / DATASET_PATH
    if not path.exists():
        return None
    return _normalize_categorical_dtypes(pd.read_excel(path, sheet_name=0))


def generate_demo_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    makes = ["Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda", "Jaguar",
             "Lexus", "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac", "Porche",
             "Saab", "Saturn", "Toyota", "VW"]

    df = pd.DataFrame({
        "Month": rng.choice(months, n),
        "WeekOfMonth": rng.integers(1, 6, n),
        "DayOfWeek": rng.choice(days, n),
        "Make": rng.choice(makes, n),
        "AccidentArea": rng.choice(["Urban", "Rural"], n, p=[0.7, 0.3]),
        "DayOfWeekClaimed": rng.choice(days, n),
        "MonthClaimed": rng.choice(months, n),
        "WeekOfMonthClaimed": rng.integers(1, 6, n),
        "Sex": rng.choice(["Male", "Female"], n),
        "MaritalStatus": rng.choice(["Single", "Married", "Widow", "Divorced"], n,
                                     p=[0.35, 0.5, 0.05, 0.10]),
        "Age": rng.normal(38, 12, n).clip(16, 80).round(0),
        "Fault": rng.choice(["Policy Holder", "Third Party"], n, p=[0.6, 0.4]),
        "PolicyType": rng.choice(
            ["Sedan - All Perils", "Sedan - Collision", "Sedan - Liability",
             "Sport - All Perils", "Sport - Collision", "Sport - Liability",
             "Utility - All Perils", "Utility - Collision", "Utility - Liability"], n),
        "VehicleCategory": rng.choice(["Sedan", "Sport", "Utility"], n, p=[0.6, 0.25, 0.15]),
        "VehiclePrice": rng.choice(
            ["less than 20000", "20000 to 29000", "30000 to 39000",
             "40000 to 59000", "60000 to 69000", "more than 69000"], n),
        "PolicyNumber": np.arange(1, n + 1),
        "RepNumber": rng.integers(1, 17, n),
        "Deductible": rng.choice([300, 400, 500, 700], n),
        "DriverRating": rng.choice([1.0, 2.0, 3.0, 4.0], n),
        "Days_Policy_Accident": rng.choice(["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"], n),
        "Days_Policy_Claim": rng.choice(["none", "8 to 15", "15 to 30", "more than 30"], n),
        "PastNumberOfClaims": rng.choice(["none", "1", "2 to 4", "more than 4"], n),
        "AgeOfVehicle": rng.choice(["new", "2 years", "3 years", "4 years", "5 years",
                                    "6 years", "7 years", "more than 7"], n),
        "AgeOfPolicyHolder": rng.choice(["16 to 17", "18 to 20", "21 to 25", "26 to 30",
                                          "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"], n),
        "PoliceReportFiled": rng.choice(["Yes", "No"], n, p=[0.1, 0.9]),
        "WitnessPresent": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
        "AgentType": rng.choice(["Internal", "External"], n, p=[0.2, 0.8]),
        "NumberOfSuppliments": rng.choice(["none", "1 to 2", "3 to 5", "more than 5"], n),
        "AddressChange_Claim": rng.choice(["no change", "under 6 months", "1 year",
                                            "2 to 3 years", "4 to 8 years"], n,
                                           p=[0.7, 0.05, 0.05, 0.1, 0.1]),
        "NumberOfCars": rng.choice(["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"], n,
                                    p=[0.5, 0.3, 0.12, 0.05, 0.03]),
        "Year": rng.choice([1994, 1995, 1996], n),
        "BasePolicy": rng.choice(["Liability", "Collision", "All Perils"], n),
        "ClaimSize": rng.lognormal(9, 1, n).round(2),
    })

    score = (
        (df["Fault"] == "Third Party").astype(int) * 0.6
        + (df["PoliceReportFiled"] == "No").astype(int) * 0.4
        + (df["WitnessPresent"] == "No").astype(int) * 0.3
        + (df["PastNumberOfClaims"] == "more than 4").astype(int) * 0.8
        + rng.normal(0, 1, n)
    )
    threshold = np.quantile(score, 0.94)
    df["FraudFound_P"] = (score > threshold).astype(int)

    df.loc[rng.choice(n, 5, replace=False), "Age"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "DriverRating"] = np.nan
    return df


# =============================================================================
# 2. VALIDATION ET CHARGEMENT
# =============================================================================

def validate_schema(df: pd.DataFrame) -> list[str]:
    """Renvoie la liste des colonnes manquantes (vide si le schema est bon)."""
    return [c for c in ALL_RAW_COLS if c not in df.columns]


def load_dataframe(file) -> pd.DataFrame:
    """Charge un .xlsx ou .csv depuis un objet fichier (upload Streamlit)."""
    name = getattr(file, "name", str(file)).lower()
    df = pd.read_csv(file) if name.endswith(".csv") else pd.read_excel(file, sheet_name=0)
    return _normalize_categorical_dtypes(df)


# =============================================================================
# 3. PREPROCESSING (fidele au notebook)
# =============================================================================

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Reproduit le preprocessing du notebook.

    Returns
    -------
    X : DataFrame des variables explicatives (encodees, memes colonnes que le
        notebook : ordinal en place, one-hot avec drop_first=True)
    y : Series cible (FraudFound_P)
    warnings : liste de messages non bloquants (valeurs inconnues rencontrees,
        imputations effectuees...)
    """
    missing = validate_schema(df)
    if missing:
        raise SchemaError(
            "Colonnes manquantes dans le fichier : " + ", ".join(missing)
        )

    warnings: list[str] = []
    df = df.copy()

    # --- Imputation des colonnes numeriques brutes (identique au notebook pour
    # Age et DriverRating ; etendue par securite a TOUTE colonne numerique qui
    # contiendrait des manquants dans le fichier reel, meme si le notebook
    # d'origine n'en avait pas besoin sur son propre jeu de donnees).
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            valeur = df[col].mean()
            df[col] = df[col].fillna(valeur)
            warnings.append(
                f"{n} valeur(s) manquante(s) dans '{col}' imputee(s) par la moyenne "
                f"({valeur:.2f})."
            )

    # --- Encodage ordinal (en place, une seule colonne par variable)
    df_ordinal = pd.DataFrame(index=df.index)
    for col, mapping in ORDINAL_MAPS.items():
        inconnues = set(df[col].dropna().unique()) - set(mapping.keys())
        if inconnues:
            warnings.append(
                f"'{col}' contient des modalites inconnues {sorted(inconnues)} "
                "-> imputees par la modalite la plus frequente."
            )
        df_ordinal[col] = df[col].map(mapping)

    # --- Encodage one-hot (drop_first=True, comme le notebook)
    df_onehot = pd.get_dummies(df[ONEHOT_COLS], columns=ONEHOT_COLS, drop_first=True)
    # bool -> int (get_dummies renvoie des bool depuis pandas >= 2.0)
    df_onehot = df_onehot.astype(int)

    X = pd.concat([df[NUMERIC_COLS], df_ordinal, df_onehot], axis=1)
    y = df[TARGET].astype(int)

    # Cast explicite en float64 AVANT le filet de securite ci-dessous : une
    # colonne peut rester en dtype "object" (notamment sur une prediction a
    # une seule ligne contenant une valeur Python None), ce que les modeles
    # (XGBoost en particulier) refusent meme une fois les NaN combles --
    # fillna() seul ne reconvertit pas le dtype d'une colonne object.
    X = X.astype(float)

    # --- Filet de securite final : quelle que soit l'origine du NaN (modalite
    # ordinale inconnue ci-dessus, valeur manquante native dans une colonne non
    # anticipee, ou toute autre anomalie du fichier reel), AUCUN NaN ne doit
    # atteindre les modeles -- LogisticRegression en particulier plante sans
    # message clair sinon. On impute par le mode de la colonne (coherent avec
    # une variable ordinale/entiere) et on avertit precisement combien.
    colonnes_avec_nan = [c for c in X.columns if X[c].isna().any()]
    for col in colonnes_avec_nan:
        n = int(X[col].isna().sum())
        mode = X[col].mode(dropna=True)
        valeur = mode.iloc[0] if len(mode) else 0
        X[col] = X[col].fillna(valeur)
        warnings.append(
            f"{n} valeur(s) residuelle(s) manquante(s) dans '{col}' imputee(s) "
            f"par la modalite la plus frequente ({valeur})."
        )

    return X, y, warnings


def align_columns(X: pd.DataFrame, reference_columns: list[str]) -> pd.DataFrame:
    """Aligne les colonnes de X sur celles utilisees a l'entrainement (pour la
    prediction d'une nouvelle observation, dont l'encodage one-hot peut ne pas
    produire exactement les memes colonnes que le jeu d'entrainement)."""
    X = X.reindex(columns=reference_columns, fill_value=0)
    return X


# =============================================================================
# 4. ENTRAINEMENT DES 3 MODELES (memes hyperparametres que le notebook)
# =============================================================================

def make_models() -> dict:
    return {
        "Régression logistique": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(
            criterion="gini", n_estimators=150, max_features="sqrt",
            max_depth=10, random_state=121,
        ),
        "XGBoost": XGBClassifier(
            max_depth=15, min_child_weight=3, gamma=8,
            eval_metric="logloss", random_state=42,
        ),
    }


def upsample(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Sur-echantillonnage de la classe minoritaire (identique au notebook)."""
    data = X_train.copy()
    data[TARGET] = y_train.values
    majority = data[data[TARGET] == 0]
    minority = data[data[TARGET] == 1]
    if len(minority) == 0:
        raise ValueError("Aucun sinistre frauduleux dans l'echantillon d'entrainement : "
                          "impossible de sur-echantillonner.")
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    up = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)
    return up.drop(columns=[TARGET]), up[TARGET]


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    report = classification_report(
        y_test, y_pred, target_names=["Non frauduleux", "Frauduleux"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)
    out = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mae": mean_absolute_error(y_test, y_pred),
        "report": report,
        "confusion_matrix": cm,
    }
    if y_proba is not None:
        try:
            out["auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            out["auc"] = float("nan")
    return out


def train_all(X: pd.DataFrame, y: pd.Series, balanced: bool, test_size: float = 0.3,
              random_state: int = 42) -> dict:
    """Entraine les 3 modeles sur les donnees brutes ou ré-equilibrees.

    Returns un dict {nom_modele: {"model":..., "metrics_train":..., "metrics_test":...}}
    plus la cle "_columns" (colonnes utilisees, pour aligner une future prediction)
    et "_split" (tailles des echantillons, pour affichage).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    split_info = {
        "n_train": len(X_train), "n_test": len(X_test),
        "taux_fraude_train": float(y_train.mean()), "taux_fraude_test": float(y_test.mean()),
    }

    if balanced:
        X_train, y_train = upsample(X_train, y_train)
        split_info["n_train_apres_reequilibrage"] = len(X_train)

    results = {"_columns": list(X_train.columns), "_split": split_info}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        results[name] = {
            "model": model,
            "metrics_train": evaluate(model, X_train, y_train),
            "metrics_test": evaluate(model, X_test, y_test),
        }
    return results


# =============================================================================
# 5. PREDICTION SUR UNE NOUVELLE OBSERVATION
# =============================================================================

def predict_one(model, reference_columns: list[str], raw_input: dict) -> tuple[int, float]:
    """Preprocesse une observation unique (dict de valeurs brutes, memes noms
    de colonnes que le fichier source) et renvoie (classe predite, proba de fraude).
    """
    df_one = pd.DataFrame([raw_input])
    # PolicyNumber et FraudFound_P ne sont pas nécessaires pour une prédiction ;
    # on les ajoute en valeurs factices pour réutiliser preprocess() tel quel.
    df_one[ID_COL] = 0
    df_one[TARGET] = 0
    X_one, _, _ = preprocess(df_one)
    X_one = align_columns(X_one, reference_columns)
    proba = float(model.predict_proba(X_one)[0, 1]) if hasattr(model, "predict_proba") else None
    pred = int(model.predict(X_one)[0])
    return pred, proba
