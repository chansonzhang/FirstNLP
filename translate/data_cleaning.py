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
# @Time    : 12/7/2020 21:51
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: data_cleaning.py

import pandas as pd
import re
import string
from string import digits

def clean(lines):
    # Lowercase all characters
    lines.eng=lines.eng.apply(lambda x: x.lower())
    lines.mar=lines.mar.apply(lambda x: x.lower())

    # Remove quotes
    lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x))
    lines.mar=lines.mar.apply(lambda x: re.sub("'", '', x))
    exclude = set(string.punctuation) # Set of all special characters

    # Remove all the special characters
    lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.mar=lines.mar.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    lines.eng=lines.eng.apply(lambda x: x.translate(remove_digits))
    lines.mar = lines.mar.apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    # Remove extra spaces
    lines.eng=lines.eng.apply(lambda x: x.strip())
    lines.mar=lines.mar.apply(lambda x: x.strip())
    lines.eng=lines.eng.apply(lambda x: re.sub(" +", " ", x))
    lines.mar=lines.mar.apply(lambda x: re.sub(" +", " ", x))

    # Add start and end tokens to target sequences
    lines.mar = lines.mar.apply(lambda x : 'START_ '+ x + ' _END')
    return lines