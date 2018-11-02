# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/10/31

from sklearn.feature_extraction.text import CountVectorizer
from WordCut import word_cut_chinese


origin_text = ['我第一次学习自然语言处理真的有点慌','不要紧张一切都会好的']
WORD_SEPARATOR = ' '
word_cut_rsts = word_cut_chinese(origin_text,WORD_SEPARATOR)

print '--------分词结果-------'
for rst in word_cut_rsts:
    print rst
print '----------------------'

count_vect = CountVectorizer()
term_matrix = count_vect.fit_transform(word_cut_rsts)

vocabulary = count_vect.vocabulary_
print '--------词汇表-------'
for (key,value) in vocabulary.items():
    print('%s:%s' %(key,value))
print '----------------------'

print('词典中word\"自然语言\"的索引是%s' % vocabulary.get(u'自然语言'))

word = filter(lambda x:x[1]==6,vocabulary.items())[0][0]
print(u'词典中索引[6]对应的word是：\"%s\"' %word)
