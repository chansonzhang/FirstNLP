# -*- coding: utf-8 -*-

# Copyright 2020 Zhang, Chen. All Rights Reserved.
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
# @Time    : 12/7/2020 21:56
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: data_prep_nmt.py
import numpy as np


def get_vocab(lines):
    # Vocabulary of English
    all_eng_words = set()
    for eng in lines.eng:
        for word in eng.split():
            if word not in all_eng_words:
                all_eng_words.add(word)

    # Vocabulary of French
    all_marathi_words = set()
    for mar in lines.mar:
        for word in mar.split():
            if word not in all_marathi_words:
                all_marathi_words.add(word)

    return all_eng_words, all_marathi_words


def get_max_len(lines):
    # Max Length of source sequence
    lenght_list = []
    for l in lines.eng:
        lenght_list.append(len(l.split(' ')))
    max_length_src = np.max(lenght_list)

    # Max Length of target sequence
    lenght_list = []
    for l in lines.mar:
        lenght_list.append(len(l.split(' ')))
    max_length_tar = np.max(lenght_list)
    return max_length_src, max_length_tar


def get_index(lines):
    all_eng_words, all_marathi_words = get_vocab(lines)
    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_marathi_words))

    # Calculate Vocab size for both source and target
    num_encoder_tokens = len(all_eng_words)
    num_decoder_tokens = len(all_marathi_words)
    num_decoder_tokens += 1  # For zero padding

    # Create word to token dictionary for both source and target
    input_token_index = dict([(word, i + 1) for i, word in enumerate(input_words)])
    target_token_index = dict([(word, i + 1) for i, word in enumerate(target_words)])

    # Create token to word dictionary for both source and target
    reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
    reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

    return num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index
