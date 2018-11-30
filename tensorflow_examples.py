# -*- coding: utf-8 -*-

# Copyright 2018-2018 Zhang, Chen.
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
# @Time    : 11/30/2018 19:39
# @Author  : Zhang, Chen (chansonzhang)
# @Email   : ZhangChen.Shaanxi@gmail.com
# @FileName: tensorflow_examples.py

import numpy as np
import tensorflow as tf

import tensorflow as tf

W1 = tf.get_variable("W1", [150, 20])
b1 = tf.get_variable("b1", [20])
W2 = tf.get_variable("W2", [20, 17])
b2 = tf.get_variable("b2", [17])

lookup = tf.get_variable("W",[1,50])

c=tf.constant(3.0)
assert c.graph == tf.get_default_graph()





g1=tf.Graph()
with g1.as_default():
    v=tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer(dtype=tf.float32))

g2 = tf.Graph()
with g2.as_default():
    v=tf.get_variable("v",shape=[1],initializer=tf.ones_initializer(dtype=tf.float32))

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))

a=10
b=5
g=tf.Graph()
with g.device('/gpu:0'):
    result = a+b


