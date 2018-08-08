# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:22:33 2018

@author: HZR
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.externals import joblib


# 模型评估，交叉验证
def rmsle_cv(model, n_folds, x, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    y_train = pd.read_csv('y_train.csv')
    test = pd.read_csv('test.csv')
    old_test = pd.read_csv('all/test.csv')
    
    # Lasso
    clf_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    score_clf_lasso = rmsle_cv(clf_lasso, 5, train.values, y_train.values)
    print("Lasso: mean:{:.4f}, std:{:.4f}".format(score_clf_lasso.mean(), score_clf_lasso.std()))
    clf_lasso.fit(train.values, y_train.values)
    # “SalePrice”之前经过log(1+x)
    clf_lasso_pred = np.expm1(clf_lasso.predict(test.values))
    
    # 保存模型
    joblib.dump(clf_lasso, "train_model.m")
#    clf_lasso = joblib.load("train_model.m")
    
    # 提交结果
    submission = pd.DataFrame()
    submission['Id'] = old_test['Id']
    submission['SalePrice'] = clf_lasso_pred
    submission.to_csv('submission.csv', index=False)














































