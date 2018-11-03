# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/11/3
import numpy as np
A = np.array([0, 1, 2])
B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
print A[None,:]
print A[:, None]

X = ([[2,5,7],[6,0,3]])
mean_X = np.mean(X, axis=0)
print X
print mean_X
print X - mean_X

mean_Y = np.mean(X, axis=1)
print mean_Y
print 'mean_Y[:, None]'
print mean_Y[:, None]

mean_Y = np.expand_dims(mean_Y, axis=1)
print 'X_Y'
print mean_Y