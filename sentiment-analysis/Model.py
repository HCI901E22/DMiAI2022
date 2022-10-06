import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.sa = None

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.loss = nn.MSELoss()

        self.seq = nn.Sequential(
            nn.Linear(4, 32),
            self.activation,
            self.dropout,
            nn.Linear(32, 256),
            self.activation,
            self.dropout,
            nn.Linear(256, 1024),
            self.activation,
            self.dropout,
            nn.Linear(1024, 1024),
            self.activation,
            self.dropout,
            nn.Linear(1024, 128),
            self.activation,
            self.dropout,
            nn.Linear(128, 1)
        )

    def load(self, path):
        nltk.download('vader_lexicon')
        self.sa = SentimentIntensityAnalyzer()
        self.load_state_dict(torch.load(path))

    def forward(self, x):
        return self.seq(x)


def preprocess_data(text: str):
    sa = SentimentIntensityAnalyzer()
    result = list(sa.polarity_scores(text).values())
    return torch.tensor(result)
