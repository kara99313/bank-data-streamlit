from pathlib import Path
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Configuration de la page
# ============================================================
st.set_page_config(
    page_title="Bank Data — AfriAI Solutions",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Prédiction de souscription bancaire")
st.caption("Projet Machine Learning — AfriAI Solutions")

st.write(
    "Renseignez le profil d’un client ou importez un fichier CSV "
    "pour obtenir une prédiction de souscription à un dépôt bancaire."
)

DOSSIER = Path(__file__).resolve().parent

# Ordre repris de l’application Gradio existante.
COLONNES = [
    "age", "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign",
    "pdays", "previous", "poutcome",
]

CATEGORIELLES = [
    "job", "marital", "education", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]

NUMERIQUES = ["age", "duration", "campaign", "pdays", "previous"]

LIBELLES = {
    "job": "Profession",
    "marital": "Situation matrimoniale",
    "education": "Niveau d’études",
    "housing": "Crédit immobilier",
    "loan": "Crédit personnel",
    "contact": "Type de contact",
    "month": "Mois du contact",
    "day_of_week": "Jour du contact",
    "poutcome": "Résultat de la campagne précédente",
}


# ============================================================
# 2. Chargement depuis le fichier regroupé
# ============================================================
@st.cache_resource
def charger_objets():
    return joblib.load(DOSSIER / "list_items.joblib")


try:
    encoders, uniques, scaler, modele = charger_objets()

    if scaler.n_features_in_ != len(COLONNES):
        raise ValueError("Le normaliseur n’attend pas 14 variables.")

    if modele.n_features_in_ != len(COLONNES):
        raise ValueError("Le modèle n’attend pas 14 variables.")

    # Le dernier encodeur correspond à la cible : no / yes.
    encodeur_cible = encoders[9]
    code_yes = encodeur_cible.transform(["yes"])[0]
    position_yes = list(modele.classes_).index(code_yes)

except Exception as erreur:
    st.error(f"Impossible de charger les objets : {erreur}")
    st.stop()


with st.expander("Informations sur le modèle"):
    st.write("Algorithme :", type(modele).__name__)
    st.write("Nombre de variables d’entrée :", modele.n_features_in_)
    st.caption(
        "L’ordre des variables reprend l’application existante. "
        "Sa concordance avec l’entraînement sera vérifiée dans le notebook."
    )

st.info(
    "La durée du dernier contact doit être connue. "
    "Cette version ne permet donc pas une utilisation entièrement "
    "avant le premier appel."
)


# ============================================================
# 3. Fonction commune de préparation et de prédiction
# ============================================================
def predire(donnees):
    if donnees.empty:
        raise ValueError("Le tableau ne contient aucune ligne.")

    if donnees.columns.duplicated().any():
        raise ValueError("Le tableau contient des noms de colonnes en double.")

    manquantes = [c for c in COLONNES if c not in donnees.columns]
    if manquantes:
        raise ValueError(
            "Colonnes manquantes : " + ", ".join(manquantes)
        )

    x = donnees[COLONNES].copy()

    if x.isna().any().any():
        colonnes_vides = x.columns[x.isna().any()].tolist()
        raise ValueError(
            "Valeurs manquantes dans : " + ", ".join(colonnes_vides)
        )

    # Vérifier les variables numériques.
    for colonne in NUMERIQUES:
        x[colonne] = pd.to_numeric(x[colonne], errors="coerce")

        if not np.isfinite(x[colonne].to_numpy(dtype=float)).all():
            raise ValueError(
                f"La colonne '{colonne}' doit contenir "
                "uniquement des nombres finis."
            )

        if (x[colonne] < 0).any():
            raise ValueError(
                f"La colonne '{colonne}' contient des valeurs négatives."
            )

    # Encoder les neuf variables catégorielles.
    for i, colonne in enumerate(CATEGORIELLES):
        valeurs = x[colonne].astype(str).str.strip()
        autorisees = set(encoders[i].classes_)
        inconnues = sorted(set(valeurs) - autorisees)

        if inconnues:
            raise ValueError(
                f"Modalités inconnues dans '{colonne}' : "
                + ", ".join(inconnues[:10])
            )

        x[colonne] = encoders[i].transform(valeurs)

    # Appliquer le normaliseur sauvegardé, puis le modèle.
    x_normalise = scaler.transform(x.to_numpy(dtype=float))

    codes_predits = modele.predict(x_normalise)
    classes_predites = encodeur_cible.inverse_transform(
        np.asarray(codes_predits, dtype=int)
    )
    probabilites = modele.predict_proba(x_normalise)[:, position_yes]

    resultat = donnees.copy()
    resultat["prediction_souscription"] = [
        "Oui" if classe == "yes" else "Non"
        for classe in classes_predites
    ]
    resultat["probabilite_souscription_pct"] = np.round(
        probabilites * 100, 2
    )

    return resultat


# ============================================================
# 4. Prédiction individuelle
# ============================================================
onglet_simple, onglet_csv = st.tabs(
    ["👤 Prédiction individuelle", "📁 Prédictions par CSV"]
)

with onglet_simple:
    st.subheader("Profil du client")
    st.caption(
        "Les modalités des listes sont conservées dans leur langue "
        "d’origine pour correspondre aux données du modèle."
    )

    with st.form("formulaire_client"):
        gauche, droite = st.columns(2)
        saisie = {}

        with gauche:
            saisie["age"] = st.number_input(
                "Âge du client",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )

            for i, colonne in enumerate(CATEGORIELLES[:5]):
                saisie[colonne] = st.selectbox(
                    LIBELLES[colonne],
                    options=list(uniques[i]),
                    key=f"champ_{colonne}",
                )

        with droite:
            for i in range(5, len(CATEGORIELLES)):
                colonne = CATEGORIELLES[i]
                saisie[colonne] = st.selectbox(
                    LIBELLES[colonne],
                    options=list(uniques[i]),
                    key=f"champ_{colonne}",
                )

            saisie["duration"] = st.number_input(
                "Durée du dernier contact, en secondes",
                min_value=0,
                value=180,
                step=1,
            )

            saisie["campaign"] = st.number_input(
                "Nombre de contacts pendant la campagne actuelle",
                min_value=1,
                value=1,
                step=1,
            )

            saisie["pdays"] = st.number_input(
                "Jours depuis le contact de la campagne précédente",
                min_value=0,
                value=999,
                step=1,
                help="999 signifie que le client n’a pas été contacté auparavant.",
            )

            saisie["previous"] = st.number_input(
                "Nombre de contacts avant la campagne actuelle",
                min_value=0,
                value=0,
                step=1,
            )

        soumettre = st.form_submit_button(
            "Prédire",
            type="primary",
        )

    if soumettre:
        try:
            donnees_client = pd.DataFrame([saisie])[COLONNES]
            resultat = predire(donnees_client)

            classe = resultat.loc[0, "prediction_souscription"]
            probabilite = resultat.loc[
                0, "probabilite_souscription_pct"
            ]

            st.subheader("Résultat")
            col1, col2 = st.columns(2)

            col1.metric("Souscription prédite", classe)
            col2.metric(
                "Probabilité estimée de souscription",
                f"{probabilite:.2f} %",
            )

            st.progress(float(probabilite) / 100)
            st.caption(
                "La probabilité est une estimation du modèle, "
                "pas une certitude."
            )

        except Exception as erreur:
            st.error(f"Prédiction impossible : {erreur}")


# ============================================================
# 5. Prédictions multiples par CSV
# ============================================================
with onglet_csv:
    st.subheader("Importer plusieurs clients")

    st.write(
        "Le fichier doit contenir les 14 colonnes suivantes. "
        "Leur ordre dans le CSV est libre."
    )
    st.code(", ".join(COLONNES), language=None)

    # Exemple fictif destiné à tester le format.
    exemple = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 180,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
    }

    st.download_button(
        "Télécharger un exemple CSV fictif",
        data=pd.DataFrame([exemple])[COLONNES].to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="exemple_clients.csv",
        mime="text/csv",
    )

    fichier = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"],
    )

    if fichier is not None:
        try:
            texte = fichier.getvalue().decode("utf-8-sig")

            donnees = pd.read_csv(
                StringIO(texte),
                sep=None,
                engine="python",
            )
            donnees.columns = donnees.columns.str.strip()

            st.write("Aperçu du fichier importé")
            st.dataframe(donnees.head(10), hide_index=True)
            st.write("Nombre de clients :", len(donnees))

            if st.button(
                "Prédire pour tous les clients",
                type="primary",
            ):
                with st.spinner("Calcul des prédictions…"):
                    resultat_csv = predire(donnees)

                st.success(
                    f"Prédictions terminées pour "
                    f"{len(resultat_csv)} client(s)."
                )
                st.dataframe(resultat_csv, hide_index=True)

                st.download_button(
                    "Télécharger les résultats",
                    data=resultat_csv.to_csv(index=False).encode(
                        "utf-8-sig"
                    ),
                    file_name="predictions_bank_data.csv",
                    mime="text/csv",
                )

        except Exception as erreur:
            st.error(f"Traitement impossible : {erreur}")