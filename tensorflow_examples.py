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




a = tf.constant([[1., 2.], [3., 4.]])
print(a.shape)
b = tf.constant([[1.], [2.]])
print(b.shape)
#c = a + tf.tile(b, [1, 2])
c = a + b
print(c.shape)

a = tf.random_uniform([5, 3, 5])
b = tf.random_uniform([5, 1, 6])

# concat a and b and apply nonlinearity
tiled_b = tf.tile(b, [1, 3, 1])
c = tf.concat([a, tiled_b], 2)
print(c.shape)
d = tf.layers.dense(c, 10, activation=tf.nn.relu)
print(d.shape)

pa = tf.layers.dense(a, 10, activation=None)
pb = tf.layers.dense(b, 10, activation=None)
print(pa.shape)
print(pb.shape)
d = tf.nn.relu(pa + pb)
print(d.shape)

A = tf.ones(shape=[3,2,5])
print(A.shape)
B = tf.ones(shape=[1,2])
print(B.shape)
C=A*B[:,:,np.newaxis]
print(C.shape)


