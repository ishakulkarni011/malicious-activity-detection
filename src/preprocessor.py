import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for pkg in ('punkt', 'stopwords', 'punkt_tab'):
    nltk.download(pkg, quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ''
        text = re.sub(r'http\S+|www\S+', ' URL ', text)
        text = re.sub(r'@\w+', ' MENTION ', text)
        text = re.sub(r'#(\w+)', r' \1 ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        return ' '.join(text.split())

    def tokenize(self, text: str) -> list[str]:
        return word_tokenize(self.clean(text))

    def preprocess(self, text: str) -> str:
        tokens = self.tokenize(text)
        tokens = [self.stemmer.stem(t) for t in tokens
                  if t not in self.stop_words and len(t) > 2]
        return ' '.join(tokens)
