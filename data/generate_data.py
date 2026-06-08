"""Generate synthetic labelled social-media data for training and evaluation."""
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Vocabulary pools ──────────────────────────────────────────────────────────

SPAM_WORDS   = ['FREE','WIN','CLICK','BUY','OFFER','PROMO','DEAL','DISCOUNT',
                'LIMITED','CASH','PRIZE','GUARANTEED','ORDER NOW','SUBSCRIBE',
                'EXCLUSIVE','URGENT','ACT NOW','EARN']
NORMAL_WORDS = ['good','day','love','enjoying','happy','thinking','beautiful',
                'interesting','learning','working','exploring','amazing',
                'sharing','meeting','reading','watching','planning']
HATE_SIGNALS = ['idiot','disgusting','pathetic','worthless','terrible','worst',
                'hate','despise','repulsive','vile','filth','trash',
                'awful','horrible','abysmal','revolting','gross']
NEUTRAL_WORDS= ['everyone','people','community','group','society','world',
                'humanity','culture','tradition','history','diversity']
PLATFORMS    = ['Twitter','Facebook','Instagram','Reddit','TikTok']

URLS  = ['http://bit.ly/win123','http://tinyurl.com/deal99','http://spam.cc/offer']
TAGS  = ['#giveaway','#freebie','#win','#deal','#promo','#sale','#free','#offer']
MENTS = ['@user1234','@official_bot','@promo_account','@click_here']


def _spam_post():
    n_urls = random.randint(1, 3)
    n_tags = random.randint(2, 6)
    words  = random.choices(SPAM_WORDS, k=random.randint(3, 7))
    urls   = random.choices(URLS,  k=n_urls)
    tags   = random.choices(TAGS,  k=n_tags)
    ments  = random.choices(MENTS, k=random.randint(0, 2))
    parts  = words + urls + tags + ments
    random.shuffle(parts)
    return ' '.join(parts) + '!' * random.randint(1, 5)


def _normal_post():
    words = random.choices(NORMAL_WORDS, k=random.randint(6, 15))
    if random.random() < 0.15:
        words.append(random.choice(TAGS))
    return ' '.join(words) + random.choice(['.', '!', '?', ''])


def _hate_post(intensity=1):
    signal_words = random.choices(HATE_SIGNALS, k=random.randint(1, 3))
    target_words = random.choices(NEUTRAL_WORDS, k=random.randint(1, 2))
    filler       = random.choices(NORMAL_WORDS,  k=random.randint(2, 5))
    parts = signal_words + target_words + filler
    random.shuffle(parts)
    return ' '.join(parts) + ('!' * intensity)


def _offensive_post():
    signal_words = random.choices(HATE_SIGNALS[:8], k=1)
    filler       = random.choices(NORMAL_WORDS,     k=random.randint(4, 8))
    parts = signal_words + filler
    random.shuffle(parts)
    return ' '.join(parts)


def _fake_account():
    age = random.randint(1, 30)
    return {
        'account_age_days': age,
        'followers':        random.randint(0, 20),
        'following':        random.randint(300, 3000),
        'post_count':       random.randint(age * 10, age * 50),
        'has_bio':          random.choice([0, 0, 0, 1]),
        'has_profile_pic':  random.choice([0, 0, 1]),
        'has_location':     0,
        'username':         f'user{"".join(str(random.randint(0,9)) for _ in range(8))}',
        'bio':              '',
    }


def _real_account():
    age = random.randint(180, 2000)
    followers = random.randint(50, 5000)
    return {
        'account_age_days': age,
        'followers':        followers,
        'following':        random.randint(50, min(followers * 3, 2000)),
        'post_count':       random.randint(20, age // 2),
        'has_bio':          1,
        'has_profile_pic':  1,
        'has_location':     random.choice([0, 1]),
        'username':         random.choice(['alice_j','bob_smith','creative_dev',
                                           'jane.doe','tech_writer','photo_fan']),
        'bio':              random.choice(['Coffee lover ☕','Developer & explorer',
                                           'Sharing thoughts daily','Nature photographer']),
    }


# ── Dataset assembly ──────────────────────────────────────────────────────────

def generate_spam_dataset(n=1200):
    rows = []
    for _ in range(n // 2):
        rows.append({'text': _spam_post(),   'label': 1})
    for _ in range(n // 2):
        rows.append({'text': _normal_post(), 'label': 0})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_fake_account_dataset(n=1200):
    rows = []
    for _ in range(n // 2):
        r = _fake_account(); r['label'] = 1; rows.append(r)
    for _ in range(n // 2):
        r = _real_account(); r['label'] = 0; rows.append(r)
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_hate_dataset(n=1500):
    rows = []
    per = n // 3
    for _ in range(per):
        rows.append({'text': _normal_post(),    'label': 0})  # neutral
    for _ in range(per):
        rows.append({'text': _offensive_post(), 'label': 1})  # offensive
    for _ in range(per):
        rows.append({'text': _hate_post(2),     'label': 2})  # hate
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    generate_spam_dataset().to_csv('data/spam.csv', index=False)
    generate_fake_account_dataset().to_csv('data/fake_accounts.csv', index=False)
    generate_hate_dataset().to_csv('data/hate_speech.csv', index=False)
    print('Sample datasets written to data/')
