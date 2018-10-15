# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:29:01 2018

@author: HZR
"""

import tensorflow as tf
import nn_inference

MODEL_SAVE_PATH = 'model/'

def test(data, para):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, para.INPUT_NODE], name='x_input')
        test_feed = {x: data.test}
        y = nn_inference.inference(x, para, None, False)
        variable_averages = tf.train.ExponentialMovingAverage(para.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pred = sess.run(y, feed_dict=test_feed)
            else:
                print('No checkpoint file found')
        return pred