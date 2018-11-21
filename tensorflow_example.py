# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/13

import tensorflow as tf

W1 = tf.get_variable("W1", [150, 20])
b1 = tf.get_variable("b1", [20])
W2 = tf.get_variable("W2", [20, 17])
b2 = tf.get_variable("b2", [17])

lookup = tf.get_variable("W",[1,50])
