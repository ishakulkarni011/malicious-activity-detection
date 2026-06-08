"""Train all three detectors, evaluate, and save models + plots."""
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_data import generate_spam_dataset, generate_fake_account_dataset, generate_hate_dataset
from src.models import SpamDetector, FakeAccountDetector, HateSpeechDetector
from src.evaluator import print_report, plot_confusion_matrix, plot_feature_importance, plot_roc_curves


def train_spam():
    print('\n═══ Spam Detector ═══')
    df = generate_spam_dataset(n=1200)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    model = SpamDetector()
    model.fit(train, train['label'].values)
    y_pred  = model.predict(test)
    y_proba = model.predict_proba(test)
    print_report(test['label'], y_pred, label='Spam Detector')
    plot_confusion_matrix(test['label'], y_pred,
                          labels=['Ham', 'Spam'],
                          title='Spam — Confusion Matrix',
                          filename='spam_cm.png')
    plot_roc_curves(test['label'].values, y_proba,
                    class_labels=['Ham', 'Spam'],
                    title='Spam — ROC Curve',
                    filename='spam_roc.png')
    # Feature importance
    tfidf_names = model.fe.tfidf.get_feature_names_out().tolist()
    stat_names  = ['char_count','word_count','url_count','hashtag_count','mention_count',
                   'exclamation_count','question_count','caps_ratio','special_ratio',
                   'repeat_chars','digit_ratio','avg_word_len']
    all_names = np.array(tfidf_names + stat_names)
    fi = model.clf.feature_importances_
    plot_feature_importance(fi, all_names, top_n=20,
                            title='Spam — Top Feature Importances',
                            filename='spam_fi.png')
    model.save()
    print('  Model saved.')


def train_fake():
    print('\n═══ Fake Account Detector ═══')
    df = generate_fake_account_dataset(n=1200)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    model = FakeAccountDetector()
    model.fit(train, train['label'].values)
    y_pred  = model.predict(test)
    y_proba = model.predict_proba(test)
    print_report(test['label'], y_pred, label='Fake Account Detector')
    plot_confusion_matrix(test['label'], y_pred,
                          labels=['Real', 'Fake'],
                          title='Fake Account — Confusion Matrix',
                          filename='fake_cm.png')
    plot_roc_curves(test['label'].values, y_proba,
                    class_labels=['Real', 'Fake'],
                    title='Fake Account — ROC Curve',
                    filename='fake_roc.png')
    feat_names = ['account_age_days','followers','following','post_count',
                  'follow_ratio','posts_per_day','has_bio','has_profile_pic',
                  'has_location','bio_length','username_digit_ratio']
    plot_feature_importance(model.clf.feature_importances_,
                            np.array(feat_names), top_n=11,
                            title='Fake Account — Feature Importances',
                            filename='fake_fi.png')
    model.save()
    print('  Model saved.')


def train_hate():
    print('\n═══ Hate Speech Detector ═══')
    df = generate_hate_dataset(n=1500)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    model = HateSpeechDetector()
    model.fit(train, train['label'].values)
    y_pred  = model.predict(test)
    y_proba = model.predict_proba(test)
    print_report(test['label'], y_pred, label='Hate Speech Detector')
    plot_confusion_matrix(test['label'], y_pred,
                          labels=['Neutral', 'Offensive', 'Hate'],
                          title='Hate Speech — Confusion Matrix',
                          filename='hate_cm.png')
    plot_roc_curves(test['label'].values, y_proba,
                    class_labels=['Neutral', 'Offensive', 'Hate'],
                    title='Hate Speech — ROC Curves',
                    filename='hate_roc.png')
    model.save()
    print('  Model saved.')


if __name__ == '__main__':
    train_spam()
    train_fake()
    train_hate()
    print('\n✓ All models trained and saved to models/')
    print('✓ Evaluation plots saved to plots/')
