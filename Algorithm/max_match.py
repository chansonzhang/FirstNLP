# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/23
import jieba


def max_match(sentence, dictionary):
    if ("" == sentence):
        return []
    word = ""
    word_end = 1
    for i in range(str(sentence).__len__(), 0, -1):
        word_tmp = sentence[0:i]
        if (word_tmp in dictionary.keys()):
            word_end = i
            break

    word = sentence[0:word_end]
    remainder = sentence[word_end:]
    return [word] + max_match(remainder, dictionary)


dic = {}

dic_file = jieba.get_dict_file()
for line in dic_file:
    word, count, type = line.decode("utf-8").split()
    dic[word] = count
dic_file.close()

sentence = "我是张晨，我爱自然语言处理"
seperator="/"

print("max match cut result:")
max_match_words = seperator.join(max_match(sentence, dic))
print(max_match_words)

print("\njieba cut result:")
jieba_cut_words = seperator.join(jieba.cut(sentence,HMM=False))
print(jieba_cut_words)

print("\njieba cut result with HMM:")
jieba_cut_words = seperator.join(jieba.cut(sentence))
print(jieba_cut_words)



