# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/15

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir='../Data/bgsj/data'
train = pd.read_csv('%s/train.txt' % data_dir)

texts = list(train['text'])

dims=2
window=5
model = Word2Vec(sentences=texts,size=dims,window=window)

print(model.wv['之'])

print(model.most_similar('之',topn=5))


total = len(texts)
vec = np.zeros([total,dims])
for i,sentence in enumerate(texts):
    counts=0
    row=[0,0]
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if 0 == counts:
        print(sentence)
    else:
        vec[i,:] = row/counts


plt.figure(figsize=(8,8))
plt.axis([-3, 3, -3, 3])
colors = list(map(lambda x:'red' if 1 == x else 'blue',train['y']))
plt.scatter(vec[:,0],vec[:,1],c=colors,alpha=0.2, s=30,lw=0)
print('Word2Vec: 白话文(蓝色)与文言文(红色)')
plt.show()

