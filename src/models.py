"""Three independent classifiers: Spam, Fake Account, Hate Speech."""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessor import TextPreprocessor
from src.feature_engineer import FeatureEngineer

MODEL_DIR = Path(__file__).parent.parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)


class SpamDetector:
    """Random Forest classifier on TF-IDF + text stats."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.fe = FeatureEngineer()
        self.clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                          class_weight='balanced', random_state=42, n_jobs=-1)
        self.trained = False

    def _prep(self, df: pd.DataFrame, fit=False):
        df = df.copy()
        df['clean_text'] = df['text'].apply(self.preprocessor.preprocess)
        return self.fe.combined(df, text_col='clean_text', fit=fit)

    def fit(self, df: pd.DataFrame, y):
        X = self._prep(df, fit=True)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, df: pd.DataFrame):
        return self.clf.predict(self._prep(df))

    def predict_proba(self, df: pd.DataFrame):
        return self.clf.predict_proba(self._prep(df))

    def save(self):
        joblib.dump({'clf': self.clf, 'fe': self.fe}, MODEL_DIR / 'spam_model.pkl')

    def load(self):
        obj = joblib.load(MODEL_DIR / 'spam_model.pkl')
        self.clf = obj['clf']
        self.fe = obj['fe']
        self.trained = True


class FakeAccountDetector:
    """Gradient Boosting on account-level behavioural features."""

    def __init__(self):
        self.fe = FeatureEngineer()
        self.clf = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def _prep(self, df: pd.DataFrame, fit=False):
        feats = self.fe.account_stats(df).values.astype(float)
        return self.scaler.fit_transform(feats) if fit else self.scaler.transform(feats)

    def fit(self, df: pd.DataFrame, y):
        X = self._prep(df, fit=True)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, df: pd.DataFrame):
        return self.clf.predict(self._prep(df))

    def predict_proba(self, df: pd.DataFrame):
        return self.clf.predict_proba(self._prep(df))

    def save(self):
        joblib.dump({'clf': self.clf, 'fe': self.fe, 'scaler': self.scaler},
                    MODEL_DIR / 'fake_model.pkl')

    def load(self):
        obj = joblib.load(MODEL_DIR / 'fake_model.pkl')
        self.clf = obj['clf']
        self.fe = obj['fe']
        self.scaler = obj['scaler']
        self.trained = True


class HateSpeechDetector:
    """Calibrated LinearSVC on TF-IDF + text stats.

    Classes: 0 = neutral, 1 = offensive, 2 = hate
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.fe = FeatureEngineer()
        svc = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
        self.clf = CalibratedClassifierCV(svc, cv=3)
        self.trained = False

    def _prep(self, df: pd.DataFrame, fit=False):
        df = df.copy()
        df['clean_text'] = df['text'].apply(self.preprocessor.preprocess)
        return self.fe.combined(df, text_col='clean_text', fit=fit)

    def fit(self, df: pd.DataFrame, y):
        X = self._prep(df, fit=True)
        self.clf.fit(X, y)
        self.trained = True

    def predict(self, df: pd.DataFrame):
        return self.clf.predict(self._prep(df))

    def predict_proba(self, df: pd.DataFrame):
        return self.clf.predict_proba(self._prep(df))

    def save(self):
        joblib.dump({'clf': self.clf, 'fe': self.fe}, MODEL_DIR / 'hate_model.pkl')

    def load(self):
        obj = joblib.load(MODEL_DIR / 'hate_model.pkl')
        self.clf = obj['clf']
        self.fe = obj['fe']
        self.trained = True
