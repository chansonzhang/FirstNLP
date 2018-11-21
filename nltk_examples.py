# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/21
import nltk
from nltk import load_parser
cfg_path='grammars/book_grammars/sql0.fcfg'
nltk.data.show_cfg(cfg_path)
cp=load_parser(cfg_path,trace=3)
query = 'What cities are located in China'
trees = list(cp.parse(query.split()))
print('trees: ')
print(trees)
answer = trees[0].label()['SEM']
answer = [s for s in answer if s]
q = ' '.join(answer)
print(q)