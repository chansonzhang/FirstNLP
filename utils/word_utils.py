# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/24
def cut_word_to_chars(word: str):
    chars = []
    for i in range(word.__len__()):
        chars.append(word[i])
    return chars