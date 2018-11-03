# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/10/31

import jieba
def word_cut_chinese(corpus, word_separator):
    word_cut_results = list()
    for document in corpus:
        word_cut_results.append(word_separator.join(jieba.cut(document)))
    return word_cut_results

