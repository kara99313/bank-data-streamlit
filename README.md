# Bank Data - Projet Final ML AfriAI Solutions

**Karaboué VAKABOU** | Classification de souscription bancaire (`yes` / `no`).

- Application publique : https://bank-data-afriai.streamlit.app
- Dépôt : https://github.com/kara99313/bank-data-streamlit

Ces liens désignent le projet existant. Leur version publique n'a pas encore été mise à jour avec ce paquet ni vérifiée pour cette nouvelle version.

## Fichiers

- `Projet_Clas_Bank_Data_VAKABOU.ipynb` : notebook source complet à exécuter localement, 3 modèles, CV à 5 plis.
- `data/Bank_Data.csv` : copie de la base fournie, SHA-256 dans les métadonnées générées.
- `bank_ml.py` : préparation commune, validation des entrées, pipeline et prédictions.
- `app.py` : Streamlit, formulaire, CSV, export CSV/HTML/PDF et possibilité de recommencer.
- `reports.py`, `fonts/` : rapports et polices pour les accents.
- `requirements.txt` : versions adaptées à l'environnement local signalé ; le notebook les réécrit selon les versions réellement utilisées.
- `requirements_validation.txt` : dépendance supplémentaire pour les tests des PDF.
- `test_bank.py`, `test_interface.py` : contrôles du modèle, des exports et de l'interface.
- `exemple_bank_data_50.csv` : 50 profils issus de la base, uniquement les variables d'entrée ; ce n'est pas un échantillon indépendant pour évaluer les performances.
- `verification_reference/` : preuve d'exécution sur la machine de l'assistant, dans d'autres versions. Ne pas utiliser ses objets pour l'application locale.
- `INSTALLATION.md` : ordre d'installation et de validation.

## Démarche

La base fournie comporte 11 640 observations, 14 entrées et la cible. Un doublon exact est supprimé : il reste 11 639 lignes. Aucun profil identique avec une cible contradictoire n'a été trouvé. Le découpage stratifié avec `random_state=42` produit 9 311 lignes train, 1 164 validation et 1 164 test.

Les catégories sont encodées par LabelEncoder comme dans le cours, avec un encodeur pour chaque variable et un autre pour la cible. L'ordre artificiel est une limite. Les catégories inédites sont encodées -1 et signalées. `unknown` est une modalité explicite.

Les paramètres IQR, la préparation KNN et StandardScaler sont appris sur train dans chaque pli. Le KNN porte sur âge, durée et nombre de contacts, sur des distances standardisées, sans utiliser la cible ni les codes catégoriels. `pdays=999` est préservé ; `previous` à IQR nul n'est pas automatiquement imputé. L'imputation ne retire aucune ligne supplémentaire. Aucune adaptation n'est apprise sur le test.

Random Forest, Gradient Boosting et XGBoost utilisent chacun quatre combinaisons de paramètres et cinq plis stratifiés. Le F1 de `yes` guide la CV et le choix sur validation ; la ROC AUC départage une égalité. Les grilles compactes ne constituent pas une optimisation exhaustive. Le modèle retenu est celui du meilleur F1 de validation, sans réentraînement train+validation. Le test est évalué ensuite une seule fois.

Le notebook affiche les countplots, pairplot, heatmap, boxplots avant/après, paramètres, Accuracy/F1/Precision/Recall/ROC AUC, F1 train, écarts train-validation, matrices, ROC, importances et diagrammes comparatifs. Il montre dix prédictions test avec classes en clair.

## Sauvegardes créées par le notebook

Dans `artifacts/` :

- `bank_pipeline.joblib` : objet complet utilisé par Streamlit.
- `preprocessor.joblib` : encodeurs et traitement IQR/KNN appris.
- `encoders.joblib`, `uniques.joblib`, `scaler.joblib`, `model.joblib`.
- `list_items.joblib = [encoders, uniques, scaler, modèle]` conformément au PDF.
- `metadata.json` : modèle retenu, paramètres, métriques, versions, date UTC, empreintes et tailles des ensembles.

**Le regroupement list_items seul ne contient pas IQR/KNN.** Pour reproduire la préparation complète, utiliser le pipeline ou le préprocesseur séparé, puis scaler et modèle. Le notebook vérifie l'égalité des classes et des probabilités après rechargement sur tout le test. Les anciens objets à la racine ne sont ni chargés par la nouvelle application ni écrasés par le notebook.

Les résultats et graphiques sont enregistrés dans `results/`. L'application charge exclusivement les nouvelles sauvegardes de `artifacts/`. Elle refuse les versions scikit-learn/XGBoost différentes des métadonnées.

## Lancement (PowerShell, .venv existant actif)

Après exécution entière du notebook :

```powershell
python -m streamlit run app.py
```

Le formulaire comporte les 14 entrées. Les CSV peuvent utiliser virgule, point-virgule ou tabulation et un ordre de colonnes libre. Valeurs manquantes, nombres invalides et colonnes dupliquées sont rejetés. Les catégories inédites et valeurs traitées sont signalées dans le résultat. Limite de l'interface : 1 000 lignes par lot pour maîtriser le coût du KNN et des rapports.

Les résultats persistent lors des téléchargements. Un changement de fichier ou de modèle invalide les anciens résultats. Un rapport comporte référence, date UTC, données, résultats, traitements, probabilités, modèle, métriques test et limites. Le PDF répartit les colonnes en plusieurs tableaux paginés avec un numéro de ligne commun.

## Limites

`duration` doit être connue après le contact : ce modèle ne convient pas au ciblage avant appel. Les probabilités sont non calibrées. La provenance précise et la représentativité de la base restent à documenter. La validation aléatoire ne constitue pas un backtesting temporel. Les importances par impureté ne démontrent pas de causalité. L'âge et d'autres caractéristiques demanderaient une analyse d'équité et un cadrage métier avant une utilisation réelle.

Usage pédagogique, sans décision bancaire automatique et sans certification de conformité revendiquée.

## Publication et remise

Après validation locale : publier le notebook **exécuté**, les modules, les nouvelles sauvegardes, `results/`, les dépendances et la documentation ; ne pas publier `.venv/` ni les archives locales. Vérifier l'application publique en navigation privée : formulaire, CSV et exports. Les résultats des tests de référence ne sont pas les tests de votre exécution locale.

La remise finale demeure le dépôt des trois notebooks exécutés et de l'URL du portfolio dans le formulaire du formateur. Échéance communiquée : 5 novembre 2026 à 23 h 59 GMT ; autorisation ultérieure de l'IA communiquée par l'apprenant.
