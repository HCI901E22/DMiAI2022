import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from Model import preprocess_data


class ReviewDataset(Dataset):
    def __init__(self, csv_path, test_train_split=0.7, train=True):
        self.data = pd.read_csv(csv_path).dropna().reset_index(drop=True)
        self.split_idx = round(test_train_split * len(self.data))
        self.data = self.data[:self.split_idx] if train else self.data[self.split_idx:]
        self.data = self.data.reset_index(drop=True)
        if len(self.data.columns) == 2:
            vader = np.asarray([preprocess_data(r) for r in self.data['reviewText']])
            self.data['neg'] = vader[::, 0]
            self.data['neu'] = vader[::, 1]
            self.data['pos'] = vader[::, 2]
            self.data['comp'] = vader[::, 3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return torch.tensor([self.data['neg'][idx], self.data['neu'][idx], self.data['pos'][idx], self.data['comp'][idx]]), self.data['overall'][idx]


class Trainer:

    def __init__(self, model, path, batchsize):

        self.model = model
        self.train_data = DataLoader(ReviewDataset(path, train=True), batch_size=batchsize, shuffle=True)
        self.test_data = DataLoader(ReviewDataset(path, train=False), batch_size=batchsize, shuffle=True)

    def train(self, lr, epochs):
        loss_fn = self.model.loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train_loop(self.model, loss_fn, optimizer)
            self.test_loop(self.model, loss_fn)
        print("Done!")

    def train_loop(self, model, loss_fn, optimizer):
        size = len(self.train_data.dataset)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for batch, (X, y) in enumerate(self.train_data):
            # Compute prediction and loss
            X = X.to(device, torch.float)
            y = y.to(device, torch.float)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current/size * 100:>0.1f}%]", end='\r')
        print()

    def test_loop(self, model, loss_fn):
        size = len(self.test_data.dataset)
        num_batches = len(self.test_data)
        test_loss, correct = 0, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            for X, y in self.test_data:
                X = X.to(device, torch.float)
                y = y.to(device, torch.float)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.4f}%, Avg loss: {test_loss:>8f} \n")
