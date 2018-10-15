# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:28:52 2018

@author: HZR
"""

import os
import numpy as np
import tensorflow as tf
import nn_inference

MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'

def train(data, para, train_data_index):
    tf.reset_default_graph()
    if train_data_index is not None:
        td = data.train[train_data_index]
        tdl = data.label[train_data_index]
    else:
        td = data.train
        tdl = data.label
    x = tf.placeholder(tf.float32, [None, para.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, para.OUTPUT_NODE], name='y_input')
    if para.REGULARIZATION_RATE == 0.0:
        y = nn_inference.inference(x, para, None, True)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(para.REGULARIZATION_RATE)
        y = nn_inference.inference(x, para, regularizer, True)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(para.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    loss = tf.sqrt(tf.losses.mean_squared_error(y, y_)) + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(para.LEARNING_RATE_BASE, 
                                               global_step, 
                                               td.shape[0]/para.BATCH_SIZE, 
                                               para.LEARNING_RATE_DECAY, 
                                               staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        np.random.seed(11)
        for i in range(para.EPOCH):
            permutation = list(np.random.permutation(td.shape[0]))
            for j in range(int(np.ceil(td.shape[0]/para.BATCH_SIZE))):
                si = j * para.BATCH_SIZE
                ei = min(si+para.BATCH_SIZE, td.shape[0])
                xs = td[permutation[si:ei], :]
                ys = tdl[permutation[si:ei]]
                _, batch_loss, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)