"""Flask web app — analyse text or images for spam, fake accounts, and hate speech."""
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd

from src.models import SpamDetector, FakeAccountDetector, HateSpeechDetector
from src import ocr_extractor

app = Flask(__name__)
CORS(app)

# ── Load models (train first if not present) ───────────────────────────────────
spam_model  = SpamDetector()
fake_model  = FakeAccountDetector()
hate_model  = HateSpeechDetector()

def _load_all():
    model_dir = Path('models')
    if not (model_dir / 'spam_model.pkl').exists():
        print('Models not found — running train.py first …')
        import train as tr
        tr.train_spam(); tr.train_fake(); tr.train_hate()
    spam_model.load()
    fake_model.load()
    hate_model.load()
    print('✓ All models loaded.')

_load_all()

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get('/')
def index():
    return render_template('index.html', ocr_available=ocr_extractor.ocr_available())


@app.post('/analyse')
def analyse():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()

    # OCR path: image_b64 supplied
    ocr_text = ''
    if data.get('image_b64'):
        ocr_text = ocr_extractor.extract_from_base64(data['image_b64'])
        if ocr_text and not text:
            text = ocr_text

    if not text:
        return jsonify({'error': 'No text or image provided.'}), 400

    row = pd.DataFrame([{'text': text}])

    # ── Spam ────────────────────────────────────────────────────────────────────
    spam_proba = spam_model.predict_proba(row)[0]
    spam_pred  = int(spam_proba[1] >= 0.5)

    # ── Hate speech ─────────────────────────────────────────────────────────────
    hate_proba = hate_model.predict_proba(row)[0]
    hate_pred  = int(hate_proba.argmax())

    # ── Fake account: derive lightweight proxy features from text ───────────────
    url_count  = text.count('http')
    tag_count  = text.count('#')
    # Build a minimal account-feature row
    account_row = pd.DataFrame([{
        'account_age_days': 1,
        'followers':        0,
        'following':        url_count * 100,   # heuristic proxy
        'post_count':       tag_count * 10,
        'has_bio':          0,
        'has_profile_pic':  0,
        'has_location':     0,
        'username':         'user' + '1234' * (url_count + 1),
        'bio':              text[:50],
    }])
    fake_proba = fake_model.predict_proba(account_row)[0]
    fake_pred  = int(fake_proba[1] >= 0.5)

    HATE_LABELS = ['Neutral', 'Offensive', 'Hate']

    return jsonify({
        'text':     text,
        'ocr_text': ocr_text,
        'spam': {
            'label':       'Spam' if spam_pred else 'Ham',
            'flagged':     bool(spam_pred),
            'probability': round(float(spam_proba[1]), 3),
        },
        'fake_account': {
            'label':       'Fake' if fake_pred else 'Real',
            'flagged':     bool(fake_pred),
            'probability': round(float(fake_proba[1]), 3),
        },
        'hate_speech': {
            'label':       HATE_LABELS[hate_pred],
            'flagged':     hate_pred > 0,
            'probability': round(float(hate_proba[hate_pred]), 3),
            'class_probs': {l: round(float(p), 3) for l, p in zip(HATE_LABELS, hate_proba)},
        },
        'overall_flagged': bool(spam_pred or fake_pred or hate_pred > 0),
    })


@app.post('/analyse-image')
def analyse_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
    img_bytes = request.files['image'].read()
    text = ocr_extractor.extract_from_bytes(img_bytes)
    if not text:
        return jsonify({'error': 'No text detected in image.'}), 422
    row = pd.DataFrame([{'text': text}])
    spam_proba = spam_model.predict_proba(row)[0]
    hate_proba = hate_model.predict_proba(row)[0]
    HATE_LABELS = ['Neutral', 'Offensive', 'Hate']
    return jsonify({
        'extracted_text': text,
        'spam':       {'flagged': bool(spam_proba[1] >= 0.5), 'probability': round(float(spam_proba[1]), 3)},
        'hate_speech':{'flagged': bool(hate_proba.argmax() > 0),
                       'label': HATE_LABELS[int(hate_proba.argmax())],
                       'probability': round(float(hate_proba.max()), 3)},
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)
