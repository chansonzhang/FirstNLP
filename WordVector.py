# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/10/31

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from WordCut import word_cut_chinese

WORD_SEPARATOR = ' '

corpus = ['我第一次学习自然语言处理真的有点慌真的好着急',
          '不要紧张一切都会好的']
word_cuted_corpus = word_cut_chinese(corpus, WORD_SEPARATOR)

print '--------分词结果-------'
for doc in word_cuted_corpus:
    print doc
print '----------------------'

count_vectorizer = CountVectorizer()
#训练
count_matrix = count_vectorizer.fit_transform(word_cuted_corpus)

tokens = count_vectorizer.vocabulary_
print '--------语料库的tokens----------'
for (token, token_index) in tokens.items():
    print('%s:%s' % (token, token_index))
print '----------------------'

print('token\"自然语言\"的索引是%s' % tokens.get(u'自然语言'))

token = filter(lambda x: x[1] == 6, tokens.items())[0][0]
print(u'索引为6的token是：\"%s\"' % token)

print '-------Count矩阵密集表示----'
print count_matrix.todense()
print '----------------------'

print '-------Count矩阵稀疏表示----'
print count_matrix
print '----------------------'

#应用到新的语料
corpus_new=['我真的很喜欢你']
word_cuted_corpus_new=word_cut_chinese(corpus_new,WORD_SEPARATOR)
matrix_new = count_vectorizer.transform(word_cuted_corpus_new)
print matrix_new.toarray()

tf_idf_transformer = TfidfTransformer()
tf_idf_matrix = tf_idf_transformer.fit_transform(count_matrix)
print '-------tf-idf矩阵密集表示----'
print tf_idf_matrix.todense()
print '----------------------'

print '-------tf-idf矩阵稀疏表示----'
print tf_idf_matrix
print '----------------------'

