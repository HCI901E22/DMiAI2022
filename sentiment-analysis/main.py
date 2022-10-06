import torch
from Model import Model
from Trainer import Trainer

print("Initializing model")
model = Model().to("cuda")

print("Loading datasets")
t = Trainer(model, "data/OfficeProducts.csv", 64)
print("Begin training")
t.train(1e-3, 64)
