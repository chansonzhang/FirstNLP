# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/23


def max_match(sentence, dictionary):
    if ("" == sentence):
        return []
    word_end = 1
    for i in range(str(sentence).__len__(), 0, -1):
        word_tmp = sentence[0:i]
        if (word_tmp in dictionary.keys()):
            word_end = i
            break

    word = sentence[0:word_end]
    remainder = sentence[word_end:]
    return [word] + max_match(remainder, dictionary)






