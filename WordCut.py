# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Data 2018/10/31

import jieba
def word_cut_chinese(origin_text,word_separator):
    word_cut_results = list()
    for sentence in origin_text:
        word_cut_results.append(word_separator.join(jieba.cut(sentence)))
    return word_cut_results

