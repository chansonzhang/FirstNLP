# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/21
import nltk
from nltk import load_parser
sentences=['fall leaves fall',
           'fall leaves fall and spring leaves spring',
           'the fall leaves left',
           'the purple dog left',
           'the dog and cat left']
cfg_path='../grammars/cs271/final_question_20.fcfg'
parser=load_parser(cfg_path)

# sentence = sentences[2]
# trees= list(parser.parse(sentence.split()))
# print('%s:%s'%(sentence, trees.__len__()))
# i=1
# print(trees)
# for tree in trees:
#     print('tree_%s:'%i)
#     print(tree)
#     i+=1

for sentence in sentences:
    trees= list(parser.parse(sentence.split()))
    print('%s:%s'%(sentence, trees.__len__()))
    # i=1
    # for tree in trees:
    #     print('tree_%s:'%i)
    #     print(tree)
    #     i+=1
