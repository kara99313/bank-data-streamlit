# Installation progressive dans le dossier existant

Ne pas créer un nouvel environnement et ne pas effacer les anciens objets.

## 1. Extraire

Extraire ce ZIP dans un dossier temporaire distinct de `Bank_Data_Streamlit`. Ne pas extraire immédiatement par-dessus les fichiers du projet. Le dossier `verification_reference` est réservé aux preuves ; ses modèles ne doivent pas être recopiés dans `artifacts/` du projet.

## 2. Sauvegarder les fichiers qui seront remplacés

Dans PowerShell, depuis le dossier existant `Bank_Data_Streamlit` :

```powershell
$bankBackup = Join-Path (Split-Path (Get-Location) -Parent) ("archives_bank_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $bankBackup
Get-ChildItem -File -Path app.py,README.md,requirements.txt,.gitignore -ErrorAction SilentlyContinue | Copy-Item -Destination $bankBackup
```

Cette archive est à côté du dépôt. Les anciens `.joblib` restent à la racine, sans modification. Si un notebook vide du même nom a déjà été créé, le conserver dans cette archive avant remplacement.

## 3. Copier les fichiers du paquet

Copier à la racine du projet : notebook, `bank_ml.py`, `app.py`, `reports.py`, `README.md`, `requirements.txt`, `requirements_validation.txt`, les deux tests, l'exemple CSV et `fonts/`. Le CSV de `data/` est celui du projet fourni ; ne le remplacer que si son empreinte diffère et après examen. Ajouter les exclusions de `GITIGNORE_A_AJOUTER.txt` au `.gitignore` existant, sans l'effacer.

Conserver `verification_reference/` à part, hors du dépôt. Il n'y a volontairement aucun dossier `artifacts/` actif dans le paquet : le notebook doit générer les objets avec votre environnement.

## 4. Dépendances

Avec le `.venv` existant actif, les installations déjà réalisées ne sont pas à refaire. Ajouter uniquement les dépendances manquantes pour les rapports/tests :

```powershell
python -m pip install reportlab==4.4.9 pypdf
python -m pip check
```

Ne pas installer les versions de `verification_reference/requirements_reference.txt` dans votre `.venv`. Le `requirements.txt` principal reprend les versions de votre environnement déjà signalées ; le notebook le régénère à partir des versions réelles. Pas de mise à jour globale de pip ni de scikit-learn requise.

## 5. Exécuter le notebook

Ouvrir `Projet_Clas_Bank_Data_VAKABOU.ipynb` dans VS Code. Sélectionner le noyau correspondant à `.venv\Scripts\python.exe` de Bank_Data_Streamlit. Exécuter les cellules du début à la fin et enregistrer le notebook avec toutes les sorties. La durée dépend du processeur, du KNN et des 60 ajustements de CV (plus réajustements sur train).

Le résultat local peut différer de la référence : retenir les résultats et objets de cette exécution, puis contrôler leur cohérence. Ne pas imposer XGBoost parce que la référence le sélectionne.

Le notebook écrit `artifacts/`, `results/` et actualise `requirements.txt`. Chaque réexécution remplace les nouvelles sauvegardes de ces deux dossiers ; archiver une exécution locale validée avant de la remplacer. Les objets historiques à la racine restent intacts.

## 6. Contrôler et lancer

Dans PowerShell :

```powershell
python -m unittest -v test_bank.py test_interface.py
python -m streamlit run app.py
```

Tester une prédiction, le CSV de 50 profils, les trois téléchargements, une nouvelle prédiction et un nouveau lot. Ouvrir les PDF/HTML et vérifier modèle, effectif, données et résultats. Pour les tests, `pypdf` est nécessaire ; il n'est pas requis par l'application.

## 7. Publier après validation

Contrôler `git status`, les fichiers suivis, l'absence d'archives et de `.venv`, puis publier les résultats locaux et les dépendances générées. Vérifier ensuite les parcours publics en navigation privée. Ne pas présenter les tests AppTest comme une vérification du navigateur public.
