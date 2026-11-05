"""Préparation commune au notebook et à Streamlit. Aucune exécution à l'import."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

FEATURES = ['age','job','marital','education','housing','loan','contact','month',
            'day_of_week','duration','campaign','pdays','previous','poutcome']
CATEGORICAL = ['job','marital','education','housing','loan','contact','month','day_of_week','poutcome']
NUMERIC = ['age','duration','campaign','pdays','previous']
# pdays=999 est un code métier ; previous a un IQR nul dans cette base.
IQR_COLUMNS = ['age','duration','campaign']
LABELS = {0: 'Non', 1: 'Oui'}


def validate_input(data):
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError('Le tableau doit contenir au moins une ligne.')
    if data.columns.duplicated().any():
        raise ValueError('Noms de colonnes en double.')
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError('Colonnes manquantes : ' + ', '.join(missing))
    x = data[FEATURES].copy()
    if x.isna().any().any():
        raise ValueError('Les valeurs manquantes doivent être renseignées.')
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors='coerce')
        v = x[col].to_numpy(dtype=float)
        if not np.isfinite(v).all() or (v < 0).any() or (v != np.floor(v)).any():
            raise ValueError(f'{col} : nombres entiers positifs ou nuls attendus.')
    if (x.age < 17).any() or (x.age > 120).any() or (x.campaign < 1).any():
        raise ValueError('Âge attendu entre 17 et 120 ans ; campaign doit être au moins 1.')
    for col in CATEGORICAL:
        x[col] = x[col].astype(str).str.strip()
        if x[col].eq('').any():
            raise ValueError(f'{col} : catégorie vide.')
    return x


class BankPreprocessor(BaseEstimator, TransformerMixin):
    """LabelEncoder par colonne, IQR + KNN sur trois mesures, sans utiliser y.

    Les objets sont ajustés exclusivement par fit, donc dans chaque pli de CV.
    Les modalités inédites reçoivent -1 ; l'application les signale.
    """
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def _mask(self, X):
        values = X[IQR_COLUMNS].astype(float).copy()
        return values.mask((values < self.lower_) | (values > self.upper_))

    def fit(self, X, y=None):
        X = validate_input(X)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        self.encoders_ = [LabelEncoder().fit(X[c]) for c in CATEGORICAL]
        self.uniques_ = [e.classes_.copy() for e in self.encoders_]
        q1, q3 = X[IQR_COLUMNS].quantile(.25), X[IQR_COLUMNS].quantile(.75)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - 1.5*iqr, q3 + 1.5*iqr
        # Un IQR nul ne justifie pas de supprimer toutes les valeurs non modales.
        self.lower_ = self.lower_.mask(iqr.eq(0), -np.inf)
        self.upper_ = self.upper_.mask(iqr.eq(0), np.inf)
        masked = self._mask(X)
        if masked.isna().all().any():
            raise ValueError('Une variable entière serait supprimée par IQR.')
        self.knn_scaler_ = StandardScaler().fit(masked)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.knn_scaler_.transform(masked))
        return self

    def transform(self, X):
        X = validate_input(X)
        out = X.copy()
        for col, encoder in zip(CATEGORICAL, self.encoders_):
            mapping = {s:i for i,s in enumerate(encoder.classes_)}
            out[col] = X[col].map(mapping).fillna(-1).astype(float)
        filled = self.imputer_.transform(self.knn_scaler_.transform(self._mask(X)))
        out[IQR_COLUMNS] = self.knn_scaler_.inverse_transform(filled)
        return out[FEATURES].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES, dtype=object)

    def flags(self, X):
        X = validate_input(X)
        flags = [[] for _ in range(len(X))]
        mask = self._mask(X).isna()
        for j, (_, row) in enumerate(mask.iterrows()):
            flags[j].extend(f'{c} : IQR/KNN' for c in IQR_COLUMNS if row[c])
        for c,e in zip(CATEGORICAL,self.encoders_):
            for j,v in enumerate(X[c]):
                if v not in e.classes_:
                    flags[j].append(f'{c} : catégorie inconnue, code -1')
        return [' ; '.join(f) if f else 'Aucun' for f in flags]


def make_pipeline(model, memory=None):
    return Pipeline([('prepare',BankPreprocessor()),('scaler',StandardScaler()),('model',model)],memory=memory)


def load_bundle(root):
    root = Path(root)
    metadata = json.loads((root/'artifacts/metadata.json').read_text(encoding='utf-8'))
    import sklearn, xgboost
    for name,current in [('scikit-learn',sklearn.__version__),('xgboost',xgboost.__version__)]:
        if current != metadata['versions'][name]:
            raise ValueError(f"Version {name} attendue : {metadata['versions'][name]}, installée : {current}.")
    artifact=root/'artifacts/bank_pipeline.joblib'
    expected=metadata['artifact_sha256']['bank_pipeline.joblib']
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected:
        raise ValueError('Le pipeline ne correspond pas aux métadonnées. Réexécuter le notebook.')
    pipe = joblib.load(artifact)
    return pipe, metadata


def predict(data, pipeline):
    x = validate_input(data)
    result = data.copy()
    predicted = pipeline.predict(x)
    position = list(pipeline.classes_).index(1)
    result['prediction_souscription'] = [LABELS[int(v)] for v in predicted]
    result['probabilite_souscription_pct'] = np.round(np.asarray(pipeline.predict_proba(x)[:,position],dtype=float)*100,2)
    result['traitements_signales'] = pipeline.named_steps['prepare'].flags(x)
    return result
