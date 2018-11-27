# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/26
"""
Reference: http://sofasofa.io/tutorials/word2vec_classifier/
Dataset: http://sofasofa.io/competitions/5/data.zip
"""
from gensim.models import Word2Vec
from sklearn.tree import DecisionTreeClassifier

from apps.biangushijin.utils import load_data_set,get_avg_word_vec_for_each_sentence,load_submit_csv

data_set=load_data_set()
train=data_set.train_data
test=data_set.test_data

n_total = len(train) + len(test)
n_train = len(train)
texts = list(train['text']) + list(test['text'])

print("begin train the word2vec model")
ndims = 100
model = Word2Vec(sentences=texts, size=ndims)

vecs = get_avg_word_vec_for_each_sentence(texts,model,ndims)

print("begin train the classifier")
clf = DecisionTreeClassifier(max_depth=3, random_state=100)

clf.fit(vecs[:n_train], train['y'])

print("begin predict")
submit=load_submit_csv()
predict_p=clf.predict_proba(vecs[n_train:])
#predict_p[:, 0]表示属于class 0（白话文）的概率
#predict_p[:, 1]表示属于class 1（古文）的概率
submit['y'] = predict_p[:, 1]
submit.to_csv('my_prediction.csv', index=False)
test['pred'] = (submit['y'] > 0.5).astype(int)
print(test.head(20))