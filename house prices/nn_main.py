# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 14:30:43 2018

@author: HZR
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
import nn_train
import nn_val
import nn_test

LAYER1_NODE = 128
LAYER2_NODE = 64
LAYER3_NODE = 32
OUTPUT_NODE = 1

EPOCH = 50
BATCH_SIZE = 20
LEARNING_RATE_BASE = 0.02
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.01
MOVING_AVERAGE_DECAY = 0.99
KEEP_PROP = 0.5
K = 5

FILENAME1 = 'EBL.txt'
FILENAME2 = 'RK.txt'

class dataStruct:
    def __init__(self, train, label, test):
        self.train = train.as_matrix()
        self.label = label.as_matrix()
        self.test = test.as_matrix()

class paraStruct:
    def __init__(self, INPUT_NODE, LAYER1_NODE, LAYER2_NODE, LAYER3_NODE, OUTPUT_NODE, BATCH_SIZE, REGULARIZATION_RATE, MOVING_AVERAGE_DECAY, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCH, KEEP_PROP, K):
        self.INPUT_NODE = INPUT_NODE
        self.LAYER1_NODE = LAYER1_NODE
        self.LAYER2_NODE = LAYER2_NODE
        self.LAYER3_NODE = LAYER3_NODE
        self.OUTPUT_NODE = OUTPUT_NODE
        self.BATCH_SIZE = BATCH_SIZE
        self.REGULARIZATION_RATE = REGULARIZATION_RATE
        self.MOVING_AVERAGE_DECAY = MOVING_AVERAGE_DECAY
        self.LEARNING_RATE_BASE = LEARNING_RATE_BASE
        self.LEARNING_RATE_DECAY = LEARNING_RATE_DECAY
        self.EPOCH = EPOCH
        self.KEEP_PROP = KEEP_PROP
        self.K = K

def k_fold_cross_valid(data, para):
    kfold = KFold(n_splits=para.K, shuffle=True, random_state=84)
    loss_sum = 0
    k = 0
    for tdi, vdi in kfold.split(data.train, data.label):
        k = k + 1
        nn_train.train(data, para, tdi)
        val_loss = nn_val.validate(data, vdi, para, k)
        loss_sum = loss_sum + val_loss
    return loss_sum / para.K

def main(argv=None):
    #--------------------------------数据读取-----------------------------------
    print('Data...')
    train_data = pd.read_csv('train.csv')
    y_train = pd.read_csv('y_train.csv')
    test_data = pd.read_csv('test.csv')
    old_test = pd.read_csv('all/test.csv')
    data = dataStruct(train_data, y_train, test_data)
    para = paraStruct(data.train.shape[1], LAYER1_NODE, LAYER2_NODE, LAYER3_NODE, OUTPUT_NODE, BATCH_SIZE, REGULARIZATION_RATE, 
                      MOVING_AVERAGE_DECAY, LEARNING_RATE_BASE, LEARNING_RATE_DECAY, EPOCH, KEEP_PROP, K)
    
    #------------------------------神经网络调参---------------------------------
    print('NN Train...')
    best_loss = float('inf')
    best_loss_1 = float('inf')
    best_para_1 = {}
    for e in range(30, 55, 5):
        for bs in range(10, 35, 5):
            for lrb in [i/1000.0 for i in range(5, 55, 5)]:
                para.EPOCH = e
                para.BATCH_SIZE = bs
                para.LEARNING_RATE_BASE = lrb
                loss = k_fold_cross_valid(data, para)
                print('epoch:{e}, batch_size:{bs}, learning_rate_base:{lrb}, loss:{l}'.format(e=e, bs=bs, lrb=lrb, l=loss))
                if loss < best_loss_1:
                    best_loss_1 = loss
                    best_para_1 = {'EPOCH':e, 'BATCH_SIZE': bs, 'LEARNING_RATE_BASE':lrb}
                    with open(FILENAME1, 'w') as f:
                        f.write('epoch:{e}, batch_size:{bs}, learning_rate_base:{lrb}, loss:{l}'.format(e=e, bs=bs, lrb=lrb, l=best_loss_1))
    
    # 'epoch':45, 'batch_size':10, 'learning_rate_base':0.01, loss:0.22074515521526336
    para.EPOCH = best_para_1['EPOCH']
    para.BATCH_SIZE = best_para_1['BATCH_SIZE']
    para.LEARNING_RATE_BASE = best_para_1['LEARNING_RATE_BASE']
#    para.EPOCH = 45
#    para.BATCH_SIZE = 10
#    para.LEARNING_RATE_BASE = 0.01
    
    best_loss_2 = float('inf')
    best_para_2 = {}
    for rr in [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        for kp in [i/10.0 for i in range(3, 11)]:
            para.REGULARIZATION_RATE = rr
            para.KEEP_PROP = kp
            loss = k_fold_cross_valid(data, para)
            print('REGULARIZATION_RATE:{rr}, KEEP_PROP:{kp}, loss:{l}'.format(rr=rr, kp=kp, l=loss))
            if loss < best_loss_2:
                best_loss_2 = loss
                best_para_2 = {'REGULARIZATION_RATE':rr, 'KEEP_PROP': kp}
                with open(FILENAME2, 'w') as f:
                    f.write('REGULARIZATION_RATE:{rr}, KEEP_PROP:{kp}, loss:{l}'.format(rr=rr, kp=kp, l=best_loss_2))
    
    # 'REGULARIZATION_RATE':0.05, 'KEEP_PROP':1.0, loss:0.1277541548013687
    para.REGULARIZATION_RATE = best_para_2['REGULARIZATION_RATE']
    para.KEEP_PROP = best_para_2['KEEP_PROP']
#    para.REGULARIZATION_RATE = 0.05
#    para.KEEP_PROP = 1.0 
    
    best_loss = best_loss_2
    
    #---------------------------神经网络训练&预测-------------------------------
    print('NN predicate...')
    nn_train.train(data, para, None)
    pred = nn_test.test(data, para)
    
    #-------------------------------输出结果------------------------------------
    print('Submission...')
    submission = pd.DataFrame()
    submission['Id'] = old_test['Id']
    submission['SalePrice'] = np.expm1(pred)
    submission.to_csv('submission.csv', index=False)
    
if __name__ == "__main__":
    tf.reset_default_graph()
    tf.app.run()