# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/24
import jieba, collections
from utils.word_utils import cut_word_to_chars


def get_jieba_dic():
    dic = {}
    dic_file = jieba.get_dict_file()
    for line in dic_file:
        word, count, type = line.decode("utf-8").split()
        dic[word] = count
    dic_file.close()
    return dic


def get_dic_of_words_with_separator(dic_of_words: dict, seperator=" "):
    "change the word in dic with space seperated,'张晨'->'张 晨'"
    dic_with_space = {}
    for key, value in dic_of_words.items():
        word_with_space = seperator.join(cut_word_to_chars(key))
        dic_with_space[word_with_space] = value
    return dic_with_space


def get_dic_of_symbols(dic_of_words, seperator=" "):
    """
    get bag of symbols
    :param dic_of_words:
    :param seperator: used to seperate symbol in word
    :return: bag of symbols, each item is key:value and key is symbol, value is symbol count
    """
    bos = collections.defaultdict(int)
    for word, count in dic_of_words.items():
        symbols = word.split(seperator)
        for s in symbols:
            bos[s] += int(count)
    return bos
