# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/15

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from apps.biangushijin.utils import load_data_set,get_average_word_vec_of_sentence,get_avg_word_vec_for_each_sentence

data_set=load_data_set()
train=data_set.train_data
test=data_set.test_data

texts = list(train['text']) + list(test['text'])

dims=2
nwindow=5
model = Word2Vec(sentences=texts, size=dims, window=nwindow)

print(model.wv['之'])
print(model.most_similar('之',topn=5))


train_size = len(train)

vec = get_avg_word_vec_for_each_sentence(train["text"],model,dims)



plt.figure(figsize=(8,8))
colors = list(map(lambda x:'red' if 1 == x else 'blue', train['y']))
plt.scatter(vec[:,0],vec[:,1],c=colors,alpha=0.2, s=30,lw=0)
print('Word2Vec: 白话文(蓝色)与文言文(红色)')
plt.show()



