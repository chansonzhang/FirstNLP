# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/25
import numpy as np


def del_cost(source_i):
    return 1


def add_cost(target_j):
    return 1


def sub_cost(source_i, target_j):
    if source_i == target_j:
        return 0
    return del_cost(source_i) + add_cost(target_j)


def min_edit_distance(source: str, target: str):
    s_len = 0
    t_len = 0
    if source:
        s_len = source.__len__()
    if target:
        t_len = target.__len__()
    if 0 == s_len and 0 == t_len:
        return 0
    D = np.zeros((s_len+1, t_len+1))
    D[0,0]=0
    for i in range(1,s_len+1):
        D[i,0]=del_cost(source[i-1])+D[i-1,0]

    for j in range(1,t_len+1):
        D[0,j]=D[0,j-1]+add_cost(target[j-1])

    for i in range(1,s_len+1):
        for j in range(1,t_len+1):
            left = D[i,j-1] + add_cost(target[j-1])
            up = del_cost(source[i-1]) + D[i-1,j]
            left_up_corner= D[i-1,j-1]+ sub_cost(source[i-1],target[j-1])
            D[i,j]=min(left,up,left_up_corner)
    #print(D)
    return D[s_len,t_len]


source = "intention"
target = "execution"
mde = min_edit_distance(source, target)
print(mde)
