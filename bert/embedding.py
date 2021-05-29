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
# @since 2021/5/29 14:32
# @author Zhang, Chen (ChansonZhang)
# @email ZhangChen.Shaanxi@gmail.com
import pandas as pd
import torch
import transformers
import numpy as np


def embed_and_save(sentences: pd.Series, file: str):
    model_class, tokenizer_class, pretrained_weights = (
        transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')

    # Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    tokenized = sentences.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

    print(padded.shape)

    attention_mask = np.where(padded != 0, 1, 0)
    print(attention_mask.shape)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states: torch.Tensor = model(input_ids, attention_mask=attention_mask).last_hidden_state

    print(last_hidden_states.shape)

    features: np.ndarray = last_hidden_states[:, 0, :].numpy()

    print(features.shape)

    features.dump(file)

    print("embedding saved at {}".format(file))
