# -*- coding: utf-8 -*-

# Copyright 2021 Zhang, Chen. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# @since 2021/5/29 13:23
# @author Zhang, Chen (ChansonZhang)
# @email ZhangChen.Shaanxi@gmail.com

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import  embedding

df: pd.DataFrame = pd.read_csv(
    #'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
    'data/SST2/train.tsv',
    delimiter='\t',
    header=None)

print(df.head())

batch_0 = df[:2000]
sentences = batch_0[0]
FEATURE_FILE='features.pkl'
#embedding.embed_and_save(sentences,FEATURE_FILE)
features = np.load(FEATURE_FILE,allow_pickle=True)
print(features.shape)

labels = batch_0[1]

train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# about 0.82
print(lr_clf.score(test_features, test_labels))





