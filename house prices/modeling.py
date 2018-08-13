# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:22:33 2018

@author: HZR
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib



# 模型评估，交叉验证
def rmsle_cv(model, n_folds, x, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

if __name__ == '__main__':
    # 读取处理后的数据
    train = pd.read_csv('train.csv')
    y_train = pd.read_csv('y_train.csv')
    test = pd.read_csv('test.csv')
    old_test = pd.read_csv('all/test.csv')
    
    # --------------------------------Lasso------------------------------------
    # 网格搜索+交叉验证
    # 'alpha':0.0005
#    clf_lasso = make_pipeline(RobustScaler(), Lasso(alpha:0.0005, random_state=1))
    best_score_lasso = float('inf');
    best_para_lasso = {};
    for a in [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]:
        print(a)
        clf_lasso = make_pipeline(RobustScaler(), Lasso(alpha=a, random_state=1))
        score_clf_lasso = rmsle_cv(clf_lasso, 5, train.values, y_train.values)
        score = score_clf_lasso.mean()
        if score < best_score_lasso:
            best_score_lasso = score
            best_para_lasso = {'alpha': a}
    
            
    clf_lasso = make_pipeline(RobustScaler(), Lasso(**best_para_lasso, random_state=1))
    clf_lasso.fit(train.values, y_train.values)
    # “SalePrice”之前经过log(1+x)
    clf_lasso_pred = np.expm1(clf_lasso.predict(test.values))
    # 模型持久化
    joblib.dump(clf_lasso, "train_model_lasso.m")
    clf_lasso = joblib.load("train_model_lasso.m")
    
    # --------------------------------ENet-------------------------------------
    # 'alpha':0.001, l1_ratio:0.5
#    clf_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=3))
    best_score_ENet = float('inf');
    best_para_ENet = {};
    for a in [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]:
        for l1r in [0.9, 0.8, 0.7, 0.6, 0.5]:
            print(a, ' ', l1r)
            clf_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=a, l1_ratio=l1r, random_state=3))
            score_clf_ENet = rmsle_cv(clf_ENet, 5, train.values, y_train.values)
            score = score_clf_ENet.mean()
            if score < best_score_ENet:
                best_score_ENet = score
                best_para_ENet = {'alpha': a, 'l1_ratio':l1r}
                
    
    clf_ENet = make_pipeline(RobustScaler(), ElasticNet(**best_para_ENet, random_state=3))
    clf_ENet.fit(train.values, y_train.values)
    clf_ENet_pred = np.expm1(clf_ENet.predict(test.values))
    joblib.dump(clf_ENet, "train_model_ENet.m")
    
    # ---------------------------------SVR-------------------------------------
     # 'C':1, 'gamma':0.01            
#    clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=1, gamma=0.01)) 
    best_score_SVR = float('inf');
    best_para_SVR = {};
    for c in [0.01, 0.5, 0.1, 1, 5, 10, 100]:
        for g in np.logspace(-2, 2, 5):
            print(c, ' ', g) 
            clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=c, gamma=g))
            score_clf_SVR = rmsle_cv(clf_SVR, 5, train.values, y_train.values.ravel())
            score = score_clf_SVR.mean()
            if score < best_score_SVR:
                best_score_SVR = score
                best_para_SVR = {'C': c, 'gamma': g}
    
               
    
    clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', **best_para_SVR))
    clf_SVR.fit(train.values, y_train.values.ravel())
    clf_SVR_pred = np.expm1(clf_SVR.predict(test.values))
    joblib.dump(clf_SVR, "train_model_SVR.m")
    
    # ---------------------------------GBR-------------------------------------
    # 'n_estimators':400, 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':20
#    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=5, max_features='sqrt', 
#                            min_samples_leaf=20, min_samples_split=50, loss='huber', random_state=5))
    best_score_GBR_1 = float('inf');
    best_para_GBR_1 = {};
    for ne in range(100, 1100, 100):
        print(ne)
        clf_GBR_1 = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=ne, learning_rate=0.1, max_depth=4, max_features='sqrt', 
                                   min_samples_leaf=15, min_samples_split=50, loss='huber', random_state=5))
        score_clf_GBR_1 = rmsle_cv(clf_GBR_1, 5, train.values, y_train.values.ravel())
        score = score_clf_GBR_1.mean()
        if score < best_score_GBR_1:
            best_score_GBR_1 = score
            best_para_GBR_1 = {'n_estimators': ne}
    
    # 'n_estimators':400
    
    best_score_GBR_2 = float('inf');
    best_para_GBR_2 = {};
    for md in range(3, 6):
        for mss in range(50, 100, 10):
            for msl in range(10, 40, 10):
                print(md, ' ', mss, ' ', msl)
                clf_GBR_2 = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=md, max_features='sqrt', 
                                          min_samples_leaf=msl, min_samples_split=mss, loss='huber', random_state=5))
                score_clf_GBR_2 = rmsle_cv(clf_GBR_2, 5, train.values, y_train.values.ravel())
                score = score_clf_GBR_2.mean()
                if score < best_score_GBR_2:
                    best_score_GBR_2 = score
                    best_para_GBR_2 = {'max_depth': md, 'min_samples_split':mss, 'min_samples_leaf':msl}
    
    # 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':20
    
    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(**best_para_GBR_1, **best_para_GBR_2, learning_rate=0.1,  max_features='sqrt', loss='huber', random_state=5))
    clf_GBR.fit(train.values, y_train.values.ravel())
    clf_GBR_pred = np.expm1(clf_GBR.predict(test.values))
    joblib.dump(clf_GBR, "train_model_GBR.m")
#    
    # ------------------------------提交结果------------------------------------
    submission = pd.DataFrame()
    submission['Id'] = old_test['Id']
    submission['SalePrice'] = clf_GBR_pred
    submission.to_csv('submission.csv', index=False)












































