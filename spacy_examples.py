# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/22

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"displaCy uses JavaScript, SVG and CSS.")
spacy.displacy.serve(doc, style='dep')