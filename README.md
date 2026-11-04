# Bank Data — Prédiction de souscription bancaire

Projet réalisé dans le cadre de la formation Machine Learning
d’AfriAI Solutions.

## Objectif

Estimer si un client va souscrire à un dépôt bancaire à partir
de son profil et des caractéristiques de la campagne.

## Fonctionnalités

- Prédiction individuelle à partir d’un formulaire.
- Affichage de la classe prédite : Oui ou Non.
- Affichage de la probabilité estimée de souscription.
- Import d’un fichier CSV contenant plusieurs clients.
- Affichage et téléchargement des résultats.
- Téléchargement d’un exemple CSV fictif.

## Modèle

L’application utilise un GradientBoostingClassifier déjà entraîné.

Le fichier `list_items.joblib` contient, dans cet ordre :

1. Les encodeurs des variables catégorielles et de la cible.
2. Les modalités des variables catégorielles.
3. Le normaliseur StandardScaler.
4. Le modèle de classification.

Les objets ont été sauvegardés avec scikit-learn 1.6.1.

## Fichiers nécessaires

- `app.py` : interface et prédictions.
- `list_items.joblib` : objets de préparation et modèle.
- `requirements.txt` : dépendances et versions.
- `.gitignore` : exclusions Git.
- `README.md` : documentation.

## Exécution locale

Python utilisé pour les essais : 3.12.3.

Installer les dépendances :

```bash
python -m pip install -r requirements.txt
```

Lancer l’application :

```bash
python -m streamlit run app.py
```

## Format du CSV

Le fichier doit être encodé en UTF-8 et contenir les colonnes :

```text
age,job,marital,education,housing,loan,contact,month,day_of_week,duration,campaign,pdays,previous,poutcome
```

L’ordre des colonnes dans le CSV est libre.
Les catégories doivent correspondre à celles proposées dans le formulaire.
Les valeurs manquantes et les catégories inconnues sont rejetées.

L’export ajoute deux colonnes :

- `prediction_souscription` : Oui ou Non.
- `probabilite_souscription_pct` : probabilité estimée, en pourcentage.

## Vérifications locales réalisées

- Chargement des objets sauvegardés.
- Essai de prédiction individuelle.
- Prédictions sur un CSV de 10 clients fictifs.
- Vérification du téléchargement des résultats.
- Résultats identiques sur ces 10 profils avant et après
  regroupement des objets.

Ces essais vérifient le fonctionnement de l’application.
Ils ne constituent pas une évaluation des performances du modèle.

## Limites

- La variable `duration` représente la durée du dernier contact :
  elle doit être connue pour effectuer une prédiction.
- Les probabilités sont des estimations du modèle.
- L’ordre des variables et les transformations reprennent
  l’application Gradio initiale ; leur concordance avec
  l’entraînement reste à vérifier dans le notebook.
- Les métriques et la justification du choix du modèle
  seront documentées après examen du notebook.