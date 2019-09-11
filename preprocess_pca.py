#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:51:41 2018

@author: bhargav
"""

import numpy as np
from sklearn.decomposition import PCA
import cv2

data = np.load('tracks_cropped_preprocessed_data.txt.npy')
data = data.astype(np.float64,copy=False)

n,h,w = data.shape

pca_res = []

for i in range(n):
    img = data[i]
    d = []
    dc = []
    for j in range(0,h,5):
        for k in range(0,w,5):
            t = img[j:j+5,k:k+5]
            t = t.flatten()
            if(np.any(t)):
                d.append(t)
                dc.append(np.array([j,k]))
    d = np.asarray(d)
    pca = PCA(n_components=1)
    pca.fit(d)
    c = pca.components_
    dt = pca.transform(d)
    residual = d - np.matmul(dt,c)
    res = np.zeros((550,200))
    p=0
    for cds in dc:
        res[cds[0]:cds[0]+5,cds[1]:cds[1]+5] = np.reshape(residual[p,:],(5,5))
        p = p+1
    res = np.asarray(res,dtype='uint8')
    pca_res.append(res)
    cv2.imwrite('tracks_pca/'+str(i+1)+'.jpg',res)
    print("%d images done"%(i+1))
pca_res = np.asarray(pca_res)
np.save('tracks_pca.txt',pca_res)