# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 17:18:35 2018

@author: HZR
"""

import tensorflow as tf

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None :
        tf.add_to_collection('losses', regularizer(weights))
    
    return weights

def inference(input_tensor, para, regularizer, dropout=True):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([para.INPUT_NODE, para.LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [para.LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer1 = tf.matmul(input_tensor, weights) + biases
        layer1 = tf.nn.relu(layer1)
        if dropout is True:
            layer1 = tf.nn.dropout(layer1, keep_prob=para.KEEP_PROP)
            
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([para.LAYER1_NODE, para.LAYER2_NODE], regularizer)
        biases = tf.get_variable('biases', [para.LAYER2_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer2 = tf.matmul(layer1, weights) + biases
        layer2 = tf.nn.relu(layer2)
        if dropout is True:
            layer2 = tf.nn.dropout(layer2, keep_prob=para.KEEP_PROP)
            
    with tf.variable_scope('layer3'):
        weights = get_weight_variable([para.LAYER2_NODE, para.LAYER3_NODE], regularizer)
        biases = tf.get_variable('biases', [para.LAYER3_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer3 = tf.matmul(layer2, weights) + biases
        layer3 = tf.nn.relu(layer3)
        if dropout is True:
            layer3 = tf.nn.dropout(layer3, keep_prob=para.KEEP_PROP)
    
    with tf.variable_scope('layer4'):
        weights = get_weight_variable([para.LAYER3_NODE, para.OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [para.OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        layer4 = tf.matmul(layer3, weights) + biases
    
    return layer4