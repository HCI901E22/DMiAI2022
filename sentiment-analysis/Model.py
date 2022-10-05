import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Model:

    def __init__(self):
        self.sa = None

    def load(self, path):
        nltk.download('vader_lexicon')
        self.sa = SentimentIntensityAnalyzer()

    def forward(self, texts):
        result = [self.sa.polarity_scores(t)['compound'] for t in texts]
        result = [((s * 5) + 5) / 2 for s in result]
        return result
