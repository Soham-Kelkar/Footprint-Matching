#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:51:01 2019

@author: sohamkelkar
"""

import torch.nn as nn
import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride, padding = 1,
                      bias=False)
    
def conv4x2(in_planes, out_planes, stride=2):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(4,2), stride=stride,
                      bias=False)
    
class BasicBlock(nn.Module):
    expansion = 1 
    
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.ReLU(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = conv4x2(1, 32)
        self.conv2 = BasicBlock(32, 32)
        self.conv3 = conv4x2(32, 64)
        self.conv4 = BasicBlock(64,512)
        self.conv5 = conv4x2(512,1024)
        self.conv6 = BasicBlock(128, 128)
        self.conv7 = conv4x2(128, 256)
        self.conv8 = BasicBlock(256, 256)
        self.ReLU = nn.ReLU(inplace=True) 
        self.classifier = nn.Linear(32, num_classes, bias = False) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def getFeatures(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ReLU(x)
        
        x = torch.mean(x, dim = 2)
        x = torch.mean(x, dim = 2)
        return x
 
    def classify(self, x):
        x = self.classifier(x)
        y = torch.norm(self.classifier.weight.data, p = 2, dim = 1).view(self.classifier.weight.data.size(0), 1)
        self.classifier.weight.data = self.classifier.weight.data/y
        return x