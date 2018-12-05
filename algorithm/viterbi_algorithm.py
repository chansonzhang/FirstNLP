# -*- coding: utf-8 -*-

# Copyright 2018 Zhang, Chen. All Rights Reserved.
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
# @Time    : 12/5/2018 11:15
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: viterbi_algorithm.py

import numpy as np
def viterbi(states:list,observations:list,PI,A,B):
    """
    the implementaion of viterbi algorithm
    :param states:
    :param observations:
    :param PI: the probability of state[i] at the very begining
    :param A: transitons probabilities, A[i][j] is the probability of transition from states[i] to states[j]
    :param B: emission probabilities, B[i][i] is the probability of seeing observations[i] given states[i]
    :return best_path_prob:
    :return best_path:list:
    """
    N=states.__len__()
    T=observations.__len__()
    viterbi_matrix=np.zeros((N,T))
    back_pointers=np.zeros(N,T)
    for s in range(0,N):
        viterbi_matrix[s][0]=PI[s]*B[s][0]
        back_pointers[s][0]=0

    for t in range(1,T):
        for s in range(0,N):
            viterbi_matrix[s][t]=viterbi_matrix[0][t-1]*A[0,s]*B[s][t]
            for pre_state in range(1,N):
                if viterbi_matrix[pre_state][t-1]*A[pre_state,s]*B[s][t] > viterbi_matrix[s][t]:
                    viterbi_matrix[s][t]=viterbi_matrix[pre_state][t-1]*A[pre_state,s]*B[s][t]
                    back_pointers[s][t]=pre_state

    best_path_prob=viterbi_matrix[0][T-1]
    best_final_state=0
    for s in range(0,N):
        if viterbi_matrix[s][T-1]>best_path_prob:
            best_final_state=s
            best_path_prob=viterbi_matrix[s][T-1]

    best_path_reverse=[best_final_state]
    current_state=best_final_state
    for t in range(T-1,0,-1):
        current_state=back_pointers[current_state][t]
        best_path_reverse.append(current_state)

    return best_path_prob,best_path_reverse.reverse()


def test_viterbi():
    """
    todo:test the viterbi algoritm
    :return:
    """



