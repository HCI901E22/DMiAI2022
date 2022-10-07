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

        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.33)
        self.loss = nn.L1Loss()
        self.scale = nn.Parameter(torch.FloatTensor([4]))

        self.seq = nn.Sequential(
            nn.Linear(4, 64),
            self.activation,
            self.dropout,
            nn.Linear(64, 512),
            self.activation,
            self.dropout,
            nn.Linear(512, 32),
            self.activation,
            self.dropout,
            # nn.Linear(1024, 1024),
            # self.activation,
            # self.dropout,
            # nn.Linear(1024, 128),
            # self.activation,
            # self.dropout,
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def load(self, path=None):
        nltk.download('vader_lexicon')
        self.sa = SentimentIntensityAnalyzer()
        if path is not None:
            self.load_state_dict(torch.load(path))

    def forward(self, x):
        return self.seq(x) * self.scale + 1


def preprocess_data(text: str):
    sa = SentimentIntensityAnalyzer()
    result = list(sa.polarity_scores(text).values())
    return result
