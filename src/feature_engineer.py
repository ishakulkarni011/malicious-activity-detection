import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer:
    def __init__(self, max_tfidf=5000):
        self.tfidf = TfidfVectorizer(
            max_features=max_tfidf,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    # ── Text statistics ────────────────────────────────────────────────────────

    def text_stats(self, texts: pd.Series) -> pd.DataFrame:
        t = texts.fillna('')
        df = pd.DataFrame()
        df['char_count']        = t.str.len()
        df['word_count']        = t.str.split().str.len().fillna(0)
        df['url_count']         = t.str.count(r'http\S+|www\S+')
        df['hashtag_count']     = t.str.count(r'#\w+')
        df['mention_count']     = t.str.count(r'@\w+')
        df['exclamation_count'] = t.str.count(r'!')
        df['question_count']    = t.str.count(r'\?')
        df['caps_ratio']        = t.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
        df['special_ratio']     = t.apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / max(len(x), 1))
        df['repeat_chars']      = t.apply(lambda x: len(re.findall(r'(.)\1{2,}', x)))
        df['digit_ratio']       = t.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
        df['avg_word_len']      = t.apply(
            lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
        )
        return df.fillna(0)

    # ── Account-level statistics (for fake-account detection) ──────────────────

    def account_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out['account_age_days']     = df.get('account_age_days', pd.Series(0, index=df.index))
        out['followers']            = df.get('followers',        pd.Series(0, index=df.index))
        out['following']            = df.get('following',        pd.Series(0, index=df.index))
        out['post_count']           = df.get('post_count',       pd.Series(0, index=df.index))
        out['follow_ratio']         = out['followers'] / (out['following'] + 1)
        out['posts_per_day']        = out['post_count'] / (out['account_age_days'] + 1)
        out['has_bio']              = df.get('has_bio',          pd.Series(0, index=df.index)).astype(int)
        out['has_profile_pic']      = df.get('has_profile_pic',  pd.Series(0, index=df.index)).astype(int)
        out['has_location']         = df.get('has_location',     pd.Series(0, index=df.index)).astype(int)
        out['bio_length']           = df.get('bio',              pd.Series('', index=df.index)).str.len().fillna(0)
        out['username_digit_ratio'] = df.get('username',         pd.Series('', index=df.index)).apply(
            lambda x: sum(c.isdigit() for c in str(x)) / max(len(str(x)), 1)
        )
        return out.fillna(0)

    # ── TF-IDF helpers ─────────────────────────────────────────────────────────

    def fit_transform_tfidf(self, texts: pd.Series):
        return self.tfidf.fit_transform(texts.fillna(''))

    def transform_tfidf(self, texts: pd.Series):
        return self.tfidf.transform(texts.fillna(''))

    # ── Combined feature matrix (TF-IDF + text stats) ─────────────────────────

    def combined(self, df: pd.DataFrame, text_col='text', fit=False):
        texts = df[text_col].fillna('')
        tfidf = self.fit_transform_tfidf(texts) if fit else self.transform_tfidf(texts)
        stats = csr_matrix(self.text_stats(df[text_col]).values)
        return hstack([tfidf, stats])
