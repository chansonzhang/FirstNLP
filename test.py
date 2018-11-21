# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/11/3
import numpy as np
A = np.array([0, 1, 2])
B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])
print(A)
print(A[None,:])
print(A[:, None])

X = ([[2,5,7],[6,0,3]])
mean_X = np.mean(X, axis=0)
print(X)
print(mean_X)
print(X - mean_X)

mean_Y = np.mean(X, axis=1)
print(mean_Y)
print('mean_Y[:, None]')
print(mean_Y[:, None])

mean_Y = np.expand_dims(mean_Y, axis=1)
print('X_Y')
print(mean_Y)

from sklearn import preprocessing
onehot_encoder = preprocessing.OneHotEncoder()
onehot_encoder.fit([[0, 0, 3],
                    [1, 1, 0],
                    [0, 2, 1],
                    [1, 0, 2]])

print(onehot_encoder.n_values_)
print(onehot_encoder.feature_indices_)
x=onehot_encoder.transform([[0, 1, 3]])

print(x)
print(x.toarray())



