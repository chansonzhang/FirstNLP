# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/24
import jieba
from algorithm.max_match import max_match
from algorithm.byte_pair_encoding import get_dic_with_bpe_symbols
from utils.corpora_utils import get_jieba_dic

dic = get_jieba_dic()

sentence = "我是张晨，我爱自然语言处理"
seperator = "/"

print("max match cut result:")
max_match_words = seperator.join(max_match(sentence, dic))
print(max_match_words)

print("\njieba cut result:")
jieba_cut_words = seperator.join(jieba.cut(sentence, HMM=False))
print(jieba_cut_words)

print("\njieba cut result with HMM:")
jieba_cut_words = seperator.join(jieba.cut(sentence, HMM=True))
print(jieba_cut_words)

dic_bpe = get_dic_with_bpe_symbols(dic)
print("max match cut(with BPE) result:")
max_match_words = seperator.join(max_match(sentence, dic_bpe))
print(max_match_words)
