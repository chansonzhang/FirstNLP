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
# @Time    : 12/7/2020 22:36
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from translate import data_cleaning, data_preprocessing, train_model, batch_generator

lines = pd.read_table('../tmp/mar.txt', names=['eng', 'mar'])
lines = data_cleaning.clean(lines)
X, y = lines.eng, lines.mar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("X_train.shape, X_test.shape", X_train.shape, X_test.shape)

X_train.to_pickle('../tmp/Weights_Mar/X_train.pkl')
X_test.to_pickle('../tmp/Weights_Mar/X_test.pkl')

num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index = data_preprocessing.get_index(lines)
max_length_src, max_length_tar = data_preprocessing.get_max_len(lines)
latent_dim = 50
model = train_model.get_model(num_encoder_tokens, num_decoder_tokens, latent_dim)

train_samples = len(X_train)
val_samples = len(X_test)
batch_size = 128
epochs = 50

model.fit_generator(
    generator=batch_generator.generate_batch(X_train, y_train, max_length_src, max_length_tar, num_decoder_tokens,
                                             input_token_index,
                                             target_token_index, batch_size=batch_size),
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    validation_data=batch_generator.generate_batch(X_test, y_test, max_length_src, max_length_tar, num_decoder_tokens,
                                                   input_token_index,
                                                   target_token_index, batch_size=batch_size),
    validation_steps=val_samples // batch_size)
