# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/26
import pandas as pd
import numpy as np
from project_root import get_project_root


class DataSet:
    train_data = None
    dev_data = None
    test_data = None


def get_data_dir():
    project_root = get_project_root()
    data_dir = project_root + '/../Data/bgsj/data'
    return data_dir


def load_data_set(path="."):
    data_dir = get_data_dir()
    train_file_path = '%s/train.txt' % data_dir
    test_file_path = '%s/test.txt' % data_dir
    print(train_file_path)
    print(test_file_path)
    data_set = DataSet()
    data_set.train_data = pd.read_csv(train_file_path, engine='python', encoding="utf-8")
    data_set.test_data = pd.read_csv(test_file_path, engine='python', encoding="utf-8")
    return data_set


def load_submit_csv():
    data_dir = get_data_dir()
    submit_file_path = '%s/sample_submit.csv' % data_dir
    submit = pd.read_csv(submit_file_path, engine='python')
    return submit


def get_average_word_vec_of_sentence(sentence, model, n_dim):
    counts = 0
    row = np.zeros(n_dim)
    for char in sentence:
        try:
            if char != ' ':
                row += model.wv[char]
                counts += 1
        except:
            pass
    if 0 == counts:
        avg_vec = np.zeros(n_dim)
        print(sentence)
    else:
        avg_vec = row / counts
    return avg_vec


def get_avg_word_vec_for_each_sentence(sentences: list, model, n_dim):
    """

    :param sentences: list of sentence，should NOT contain index
    :param model:
    :param n_dim:
    :return:
    """
    vec = np.zeros([sentences.__len__(), n_dim])
    for i, s in enumerate(sentences):
        vec[i] = get_average_word_vec_of_sentence(s, model, n_dim)
    return vec
