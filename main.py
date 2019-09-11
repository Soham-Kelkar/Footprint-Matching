#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:56:50 2019

@author: sohamkelkar
"""

import numpy as np
import torch
from ResNet import ResNet, BasicBlock
from Trainer import Trainer
from Dataset import Dataset, Testset

import pandas as pd

X = np.load('tracks_cropped_preprocessed_data.npy')
all_labels = np.load('references_preprocessed_data.npy')

labels = pd.read_csv("label_table.csv", header = None).values
y = labels[:, 1]

train_X = X[:300]
train_y = y[:300]

labels_req = np.vstack((all_labels[:100], all_labels[1000:1110]))

n_epochs = 50
batch_size = 16

train_set = Dataset(train_X, train_y)
test_set = Testset(labels_req)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers = 8, shuffle  = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers = 8, shuffle  = False)

print("ADAM OPTIMIZER")
model = ResNet(BasicBlock, 1175)
AdamOptimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
adam_trainer = Trainer(model, AdamOptimizer)
adam_trainer.run(n_epochs, train_set, train_loader)

adam_trainer.getLabelFeatures(test_loader)
results = adam_trainer.getScores(X[200:])
np.save("results_ranking.npy",np.asarray(results))