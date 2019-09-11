#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 22:45:09 2018

@author: sohamkelkar
"""

import numpy as np
import torch.nn as nn
import torch

class Trainer():
    def __init__(self, model, optimizer):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss()
        self.results = []
        self.label_features = []
        self.lr = 1e-2
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
  
    def run(self, epochs, trainset, loader):
        print("Start Training...")
        self.model.train()
        for e in range(epochs):
            epoch_loss = 0 
            correct = 0
            total = 0
            for batch_idx, (Data, label) in enumerate(loader):
                self.optimizer.zero_grad()
                X = Data.float().cuda()
                Y = label.long().cuda() 
                features = self.model.getFeatures(X)
                out = self.model.classify(features)
                pred = torch.max(out.data,1)[1] 
                predicted = (pred == Y)
                correct += predicted.sum()
                total += pred.shape[0]
                loss = self.loss(out, Y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data
            if ((e + 1) % 10 == 0):
                self.lr = self.lr/10
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay = 1e-6)
            total_loss = epoch_loss.cpu().detach().numpy()/trainset.__len__()
            print("epoch : {0}, total_loss: {1:.6f}".format(e+1, total_loss))
                
    def getScores(self, x):
        self.model.eval()
        self.optimizer.zero_grad()
        x = torch.from_numpy(x)
        for j in range(100):
            if (j % 10 == 0):
                print(j)
            scores = []
            data = x[j].unsqueeze(0).float().cuda()
            features1 = self.model.getFeatures(data).squeeze(0)
            for i in range(len(self.label_features)):
                features2 = self.label_features[i].cuda()
                scores.append(torch.nn.functional.cosine_similarity(features1, 
                                                                    features2,
                                                                    dim = -1,  
                                                                    eps=1e-08).cpu().detach()) 
            self.results.append(sorted(range(len(scores)),key=scores.__getitem__,reverse=True))
            for k in range(len(self.results[j])):
                if (self.results[j][k] > 99):
                    self.results[j][k] += 900
                self.results[j][k] += 1
        return self.results

    def getLabelFeatures(self, loader):
        self.model.eval()
        self.optimizer.zero_grad()
        for batch_idx, y in enumerate(loader):
            y = y.float().cuda()
            out = self.model.getFeatures(y)
            self.label_features.append(out.cpu().detach())

        self.label_features = torch.cat(self.label_features, dim = 0)

    def test(self, testset, loader):
        self.model.eval()
        scores = []
        for batch_idx, (x1, x2) in enumerate(loader):
            self.optimizer.zero_grad()
            x1 = x1.cuda()
            x2 = x2.cuda()
            features1 = self.model.getFeatures(x1)
            features2 = self.model.getFeatures(x2)
            if (batch_idx % 100 == 0):
                print(batch_idx, int(testset.__len__()/8))
            for i in range(len(features1)):
                scores.append(torch.nn.functional.cosine_similarity(features1[i], features2[i], dim = 0, eps=1e-08).cpu().detach())
        np.save('scores_final', np.asarray(scores))       