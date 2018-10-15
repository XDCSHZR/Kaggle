# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:28:21 2018

@author: HZR
"""

import tensorflow as tf
import nn_inference

MODEL_SAVE_PATH = 'model/'

def validate(data, validate_data_index, para, k):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, para.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, para.OUTPUT_NODE], name='y_input')
        validate_feed = {x: data.train[validate_data_index], y_: data.label[validate_data_index]}
        y = nn_inference.inference(x, para, None, False)
        loss = tf.sqrt(tf.losses.mean_squared_error(y, y_))
        variable_averages = tf.train.ExponentialMovingAverage(para.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                val_loss = sess.run(loss, feed_dict=validate_feed)
                print('fold:{f}, validation loss:{vl}'.format(f=k, vl=val_loss))
            else:
                print('No checkpoint file found')
        return val_loss