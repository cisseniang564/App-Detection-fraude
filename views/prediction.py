import streamlit as st

from pipeline import ONEHOT_COLS, ORDINAL_MAPS, predict_one
from ui import hero, inject_css, section, verdict

inject_css()

hero(
    "Prédiction",
    "Évaluer un nouveau sinistre",
    "Renseignez les caractéristiques d'un sinistre pour obtenir une probabilité de "
    "fraude avec le modèle entraîné.",
)

if "results_balanced" not in st.session_state:
    st.info("Entraînez d'abord les modèles dans l'onglet **Modélisation**.")
    st.page_link("views/modelisation.py", label="Aller à la modélisation", icon=":material/arrow_back:")
    st.stop()

df = st.session_state["df"]
res_bal = st.session_state["results_balanced"]
res_brut = st.session_state["results_brut"]
MODELS = ["Régression logistique", "Random Forest", "XGBoost"]

col_a, col_b = st.columns(2)
with col_a:
    scenario = st.selectbox("Modèle entraîné sur…", ["Données ré-équilibrées", "Données brutes"])
with col_b:
    nom_modele = st.selectbox("Algorithme", MODELS, index=2)

resultats = res_bal if scenario == "Données ré-équilibrées" else res_brut
model = resultats[nom_modele]["model"]
reference_columns = resultats["_columns"]


def options(col: str) -> list:
    return sorted(df[col].dropna().unique().tolist())


section("Caractéristiques du sinistre")
with st.form("formulaire_sinistre"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sinistre**")
        month = st.selectbox("Mois de survenance", list(ORDINAL_MAPS["Month"].keys()))
        week = st.number_input("Semaine du mois", 1, 5, 2)
        dow = st.selectbox("Jour de la semaine", list(ORDINAL_MAPS["DayOfWeek"].keys()))
        area = st.selectbox("Zone de l'accident", options("AccidentArea"))
        month_c = st.selectbox("Mois de déclaration", list(ORDINAL_MAPS["MonthClaimed"].keys()))
        week_c = st.number_input("Semaine du mois (déclaration)", 1, 5, 2)
        dow_c = st.selectbox("Jour de déclaration", list(ORDINAL_MAPS["DayOfWeekClaimed"].keys()))
        fault = st.selectbox("Responsabilité", options("Fault"))
    with c2:
        st.markdown("**Assuré**")
        sex = st.selectbox("Sexe", options("Sex"))
        marital = st.selectbox("Situation familiale", options("MaritalStatus"))
        age = st.number_input("Âge de l'assuré", 16, 90, 35)
        age_holder = st.selectbox("Ancienneté (âge du contrat)", list(ORDINAL_MAPS["AgeOfPolicyHolder"].keys()))
        addr_change = st.selectbox("Changement d'adresse récent", options("AddressChange_Claim"))
        n_cars = st.selectbox("Nombre de véhicules assurés", options("NumberOfCars"))
        police_report = st.selectbox("Rapport de police déposé", options("PoliceReportFiled"))
        witness = st.selectbox("Témoin présent", options("WitnessPresent"))
    with c3:
        st.markdown("**Police & véhicule**")
        policy_type = st.selectbox("Type de police", options("PolicyType"))
        base_policy = st.selectbox("Garantie de base", options("BasePolicy"))
        vehicle_cat = st.selectbox("Catégorie de véhicule", options("VehicleCategory"))
        vehicle_price = st.selectbox("Prix du véhicule", list(ORDINAL_MAPS["VehiclePrice"].keys()))
        make = st.selectbox("Marque", options("Make"))
        vehicle_age = st.selectbox("Âge du véhicule", list(ORDINAL_MAPS["AgeOfVehicle"].keys()))
        year = st.number_input("Année de la police", 1993, 1998, 1995)
        claim_size = st.number_input("Montant estimé du sinistre (€)", 0.0, 500000.0, 8000.0, step=500.0)

    with st.expander("Autres caractéristiques (valeurs par défaut raisonnables)"):
        d1, d2, d3 = st.columns(3)
        with d1:
            rep_number = st.number_input("Numéro du représentant", 1, 30, 10)
            deductible = st.selectbox("Franchise", sorted(df["Deductible"].unique().tolist()))
        with d2:
            driver_rating = st.selectbox("Note du conducteur", sorted(df["DriverRating"].dropna().unique().tolist()))
            days_acc = st.selectbox("Délai police→accident", list(ORDINAL_MAPS["Days_Policy_Accident"].keys()))
        with d3:
            days_claim = st.selectbox("Délai police→déclaration", list(ORDINAL_MAPS["Days_Policy_Claim"].keys()))
            past_claims = st.selectbox("Sinistres passés", options("PastNumberOfClaims"))
        d4, d5 = st.columns(2)
        with d4:
            agent_type = st.selectbox("Type d'agent", options("AgentType"))
        with d5:
            n_suppl = st.selectbox("Suppléments", options("NumberOfSuppliments"))

    submit = st.form_submit_button("Évaluer ce sinistre", type="primary", width="stretch")

if submit:
    raw = {
        "Month": month, "WeekOfMonth": week, "DayOfWeek": dow, "Make": make,
        "AccidentArea": area, "DayOfWeekClaimed": dow_c, "MonthClaimed": month_c,
        "WeekOfMonthClaimed": week_c, "Sex": sex, "MaritalStatus": marital, "Age": age,
        "Fault": fault, "PolicyType": policy_type, "VehicleCategory": vehicle_cat,
        "VehiclePrice": vehicle_price, "RepNumber": rep_number, "Deductible": deductible,
        "DriverRating": driver_rating, "Days_Policy_Accident": days_acc,
        "Days_Policy_Claim": days_claim, "PastNumberOfClaims": past_claims,
        "AgeOfVehicle": vehicle_age, "AgeOfPolicyHolder": age_holder,
        "PoliceReportFiled": police_report, "WitnessPresent": witness,
        "AgentType": agent_type, "NumberOfSuppliments": n_suppl,
        "AddressChange_Claim": addr_change, "NumberOfCars": n_cars, "Year": year,
        "BasePolicy": base_policy, "ClaimSize": claim_size,
    }
    pred, proba = predict_one(model, reference_columns, raw)
    section("Résultat")
    if proba is not None:
        verdict(proba)
    else:
        st.write("Classe prédite :", "Frauduleux" if pred else "Non frauduleux")
    st.caption(f"Modèle utilisé : {nom_modele} ({scenario.lower()}).")
