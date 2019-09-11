#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:04:58 2019

@author: sohamkelkar
"""

import torch.nn as nn
import numpy as np

def init_xavier(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)