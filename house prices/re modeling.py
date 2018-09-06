# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 22:22:33 2018

@author: HZR
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from mlxtend.regressor import StackingRegressor


# 模型评估，交叉验证, rmsle（预测结果price之前经过log）分数越小越好
def rmsle_cv(model, n_folds, x, y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x)
    rmse= np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# 模型评价，整体训练集预测准确与否
def rmsle_all(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    
    
# Stacking(base model + meta model)
class StackingModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, baseModels, metaModel, n_folds=5):
        self.baseModels = baseModels
        self.metaModel = metaModel
        self.n_folds = n_folds
    
    def fit(self, X, y):
        self.baseModels_ = [list() for x in self.baseModels]
        self.metaModel_ = clone(self.metaModel)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=84)
        out_of_fold_predicitions = np.zeros((X.shape[0], len(self.baseModels)))
        for index, model in enumerate(self.baseModels):
            for train_index, holdout_index in kfold.split(X, y):
                tempModel = clone(model)
                tempModel.fit(X[train_index], y[train_index])
                self.baseModels_[index].append(tempModel)
                out_of_fold_predicitions[holdout_index, index] = tempModel.predict(X[holdout_index])
        self.metaModel_.fit(out_of_fold_predicitions, y)
        return self
    
    def predict(self, X):
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in baseModels]).mean(axis=1) for baseModels in self.baseModels_])
        return self.metaModel_.predict(meta_features)
        
    
if __name__ == '__main__':
    # 读取处理后的数据
    print('Data...')
    train = pd.read_csv('train.csv')
    y_train = pd.read_csv('y_train.csv')
    test = pd.read_csv('test.csv')
    old_test = pd.read_csv('all/test.csv')
    
    # --------------------------------Lasso------------------------------------
    print('Lasso...')
    # 网格搜索+交叉验证
    # 'alpha':0.001, score:0.1176647811479667, train score:0.10970698333903418
    clf_Lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.001, random_state=1))
    best_score_Lasso = 0.1176647811479667
#    best_score_lasso = float('inf');
#    best_para_lasso = {};
#    for a in [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]:
#        clf_lasso = make_pipeline(RobustScaler(), Lasso(alpha=a, random_state=1))
#        score_clf_lasso = rmsle_cv(clf_lasso, 5, train.values, y_train.values.ravel())
#        score = score_clf_lasso.mean()
#        print(a, ' score:', score)
#        if score < best_score_lasso:
#            best_score_lasso = score
#            best_para_lasso = {'alpha': a}
#   
#    clf_lasso = make_pipeline(RobustScaler(), Lasso(**best_para_lasso, random_state=1))
    clf_Lasso.fit(train.values, y_train.values.ravel())
    clf_Lasso_train_pred = clf_Lasso.predict(train.values)
    clf_Lasso_train_score = rmsle_all(y_train.values.ravel(), clf_Lasso_train_pred)
    # “SalePrice”之前经过log(1+x)
    clf_Lasso_pred = np.expm1(clf_Lasso.predict(test.values))
    # 模型持久化
    joblib.dump(clf_Lasso, "train_model_lasso.m")
#    clf_lasso = joblib.load("train_model_lasso.m")
    
    # --------------------------------ENet-------------------------------------
    print('ENet...')
    # 'alpha':0.01, 'l1_ratio':0.1, score:0.1176541333679045, train score:0.11013515984679859
    clf_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=3))
    best_score_ENet = 0.1176541333679045
#    best_score_ENet = float('inf');
#    best_para_ENet = {};
#    for a in [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]:
#        for l1r in [i/10.0 for i in range(1, 10)]:
#            clf_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=a, l1_ratio=l1r, random_state=3))
#            score_clf_ENet = rmsle_cv(clf_ENet, 5, train.values, y_train.values.ravel())
#            score = score_clf_ENet.mean()
#            print(a, ' ', l1r, ' score:', score)
#            if score < best_score_ENet:
#                best_score_ENet = score
#                best_para_ENet = {'alpha': a, 'l1_ratio':l1r}
#                
#    clf_ENet = make_pipeline(RobustScaler(), ElasticNet(**best_para_ENet, random_state=3))
    clf_ENet.fit(train.values, y_train.values.ravel())
    clf_ENet_train_pred = clf_ENet.predict(train.values)
    clf_ENet_train_score = rmsle_all(y_train.values.ravel(), clf_ENet_train_pred)
    clf_ENet_pred = np.expm1(clf_ENet.predict(test.values))
    joblib.dump(clf_ENet, "train_model_ENet.m")
    
    # ---------------------------------SVR-------------------------------------
    print('SVR...')
    # 'C':1, 'gamma':0.01, score:0.2214320818153098, train score:0.08947125069713543
    clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=1, gamma=0.01)) 
    best_score_SVR = 0.2214320818153098
#    best_score_SVR = float('inf');
#    best_para_SVR = {};
#    for c in [0.01, 0.5, 0.1, 1, 5, 10, 100]:
#        for g in np.logspace(-2, 2, 5):
#            clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=c, gamma=g))
#            score_clf_SVR = rmsle_cv(clf_SVR, 5, train.values, y_train.values.ravel())
#            score = score_clf_SVR.mean()
#            print(c, ' ', g, ' score:', score) 
#            if score < best_score_SVR:
#                best_score_SVR = score
#                best_para_SVR = {'C': c, 'gamma': g}
#    
#    clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', **best_para_SVR))
    clf_SVR.fit(train.values, y_train.values.ravel())
    clf_SVR_train_pred = clf_SVR.predict(train.values)
    clf_SVR_train_score = rmsle_all(y_train.values.ravel(), clf_SVR_train_pred)
    clf_SVR_pred = np.expm1(clf_SVR.predict(test.values))
    joblib.dump(clf_SVR, "train_model_SVR.m")
    
    # ---------------------------------GBR-------------------------------------
    print('GBR...')
    # 'n_estimators':500, 'max_depth':4, 'min_samples_split':50, 'min_samples_leaf':30, score:0.11807760041186348, train score:0.08453714548732104
    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=4, max_features='sqrt', 
                            min_samples_leaf=30, min_samples_split=50, loss='huber', random_state=5))
    best_score_GBR = 0.1181680032634042
#    best_score_GBR_1 = float('inf');
#    best_para_GBR_1 = {};
#    for ne in range(100, 1100, 100):
#        clf_GBR_1 = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=ne, learning_rate=0.1, max_depth=4, max_features='sqrt', 
#                                  min_samples_leaf=15, min_samples_split=50, loss='huber', random_state=5))
#        score_clf_GBR_1 = rmsle_cv(clf_GBR_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_GBR_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_GBR_1:
#            best_score_GBR_1 = score
#            best_para_GBR_1 = {'n_estimators': ne}
#    
#    # 'n_estimators':500, score:0.11959521167349499
#    
#    best_score_GBR_2 = float('inf');
#    best_para_GBR_2 = {};
#    for md in range(3, 6):
#        for mss in range(50, 100, 10):
#            for msl in range(10, 40, 10):
#                clf_GBR_2 = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=best_para_GBR_1['n_estimators'], learning_rate=0.1, max_depth=md, max_features='sqrt', 
#                                          min_samples_leaf=msl, min_samples_split=mss, loss='huber', random_state=5))
#                score_clf_GBR_2 = rmsle_cv(clf_GBR_2, 5, train.values, y_train.values.ravel())
#                score = score_clf_GBR_2.mean()
#                print(md, ' ', mss, ' ', msl, ' score:', score)
#                if score < best_score_GBR_2:
#                    best_score_GBR_2 = score
#                    best_para_GBR_2 = {'max_depth': md, 'min_samples_split':mss, 'min_samples_leaf':msl}
#    
#    # 'max_depth':4, 'min_samples_split':50, 'min_samples_leaf':30, score:0.1181680032634042
#    
#    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(**best_para_GBR_1, **best_para_GBR_2, learning_rate=0.1,  max_features='sqrt', loss='huber', random_state=5))
#    best_score_GBR = best_score_GBR_2
    clf_GBR.fit(train.values, y_train.values.ravel())
    clf_GBR_train_pred = clf_GBR.predict(train.values)
    clf_GBR_train_score = rmsle_all(y_train.values.ravel(), clf_GBR_train_pred)
    clf_GBR_pred = np.expm1(clf_GBR.predict(test.values))
    joblib.dump(clf_GBR, "train_model_GBR.m")
    
    # ---------------------------------RFR-------------------------------------
    print('RFR...')
    # 'n_estimators':1800, 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':10, score:0.1718004443413196, train score:0.1532245299994691
    clf_RFR = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=1800, max_depth=5, max_features='sqrt', 
                            min_samples_leaf=10, min_samples_split=50, random_state=7))
    best_score_RFR = 0.1718004443413196
#    best_score_RFR_1 = float('inf');
#    best_para_RFR_1 = {};
#    for ne in range(100, 2100, 100):
#        clf_RFR_1 = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=ne, max_depth=4, max_features='sqrt', 
#                                  min_samples_leaf=15, min_samples_split=50, random_state=7))
#        score_clf_RFR_1 = rmsle_cv(clf_RFR_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_RFR_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_RFR_1:
#            best_score_RFR_1 = score
#            best_para_RFR_1 = {'n_estimators': ne}
#    
#    # 'n_estimators':1800, score:0.182019313675976
#    
#    best_score_RFR_2 = float('inf');
#    best_para_RFR_2 = {};
#    for md in range(3, 6):
#        for mss in range(50, 100, 10):
#            for msl in range(10, 40, 10):
#                clf_RFR_2 = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=best_para_RFR_1['n_estimators'], max_depth=md, max_features='sqrt', 
#                                          min_samples_leaf=msl, min_samples_split=mss, random_state=7))
#                score_clf_RFR_2 = rmsle_cv(clf_RFR_2, 5, train.values, y_train.values.ravel())
#                score = score_clf_RFR_2.mean()
#                print(md, ' ', mss, ' ', msl, ' score:', score)
#                if score < best_score_RFR_2:
#                    best_score_RFR_2 = score
#                    best_para_RFR_2 = {'max_depth': md, 'min_samples_split':mss, 'min_samples_leaf':msl}
#    
#    # 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':10, score:0.1718004443413196
#    
#    clf_RFR = make_pipeline(RobustScaler(), RandomForestRegressor(**best_para_RFR_1, **best_para_RFR_2, max_features='sqrt', random_state=7))
#    best_score_RFR = best_score_RFR_2
    clf_RFR.fit(train.values, y_train.values.ravel())
    clf_RFR_train_pred = clf_RFR.predict(train.values)
    clf_RFR_train_score = rmsle_all(y_train.values.ravel(), clf_RFR_train_pred)
    clf_RFR_pred = np.expm1(clf_RFR.predict(test.values))
    joblib.dump(clf_RFR, "train_model_RFR.m")
    
    #----------------------------------XGB-------------------------------------
    print('XGB...')
    # 'n_estimators':2000, 'max_depth':4, 'min_child_weight':3, 'gamma':0.0, 'subsample':0.6, 'colsample_bytree':0.5, 'reg_alpha':0.001, 'reg_lambda':1, score:0.11832497642250632, train score:0.0025294371702461356
    clf_XGB = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.0, learning_rate=0.1, max_depth=4, min_child_weight=3, n_estimators=2000, 
                            reg_alpha=0.001, reg_lambda=1, subsample=0.6, silent=1, random_state=9))
    best_score_XGB = 0.11832497642250632
#    best_score_XGB_1 = float('inf');
#    best_para_XGB_1 = {};
#    for ne in range(100, 3000, 100):
#        clf_XGB_1 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.1, learning_rate=0.1, max_depth=4, min_child_weight=2, n_estimators=ne, 
#                                  reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, silent=1, random_state=9))
#        score_clf_XGB_1 = rmsle_cv(clf_XGB_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_XGB_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_XGB_1:
#            best_score_XGB_1 = score
#            best_para_XGB_1 = {'n_estimators': ne}
#    
#     # 'n_estimators':2000, score:0.12320212008164114
#    best_para_XGB_1 = {'n_estimators': 2000}
#    best_score_XGB_2 = float('inf');
#    best_para_XGB_2 = {};
#    for md in range(3, 6):
#        for mcw in range(1, 6):
#            clf_XGB_2 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.1, learning_rate=0.1, max_depth=md, min_child_weight=mcw, n_estimators=best_para_XGB_1['n_estimators'], 
#                                  reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, silent=1, random_state=9))
#            score_clf_XGB_2 = rmsle_cv(clf_XGB_2, 5, train.values, y_train.values.ravel())
#            score = score_clf_XGB_2.mean()
#            print(md, ' ', mcw, ' score:', score)
#            if score < best_score_XGB_2:
#                best_score_XGB_2 = score
#                best_para_XGB_2 = {'max_depth': md, 'min_child_weight': mcw}
#    
#    # 'max_depth':4, 'min_child_weight':3, score:0.12107877034852185
#    
#    best_score_XGB_3 = float('inf');
#    best_para_XGB_3 = {};
#    for g in [i/10.0 for i in range(0, 6)]:
#        clf_XGB_3 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.8, gamma=g, learning_rate=0.1, max_depth=best_para_XGB_2['max_depth'], min_child_weight=best_para_XGB_2['min_child_weight'], n_estimators=best_para_XGB_1['n_estimators'], 
#                                  reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, silent=1, random_state=9))
#        score_clf_XGB_3 = rmsle_cv(clf_XGB_3, 5, train.values, y_train.values.ravel())
#        score = score_clf_XGB_3.mean()
#        print(g, ' score:', score)
#        if score < best_score_XGB_3:
#            best_score_XGB_3 = score
#            best_para_XGB_3 = {'gamma': g}
#    
#    # 'gamma':0.0, score:0.12095040013901592
#    
#    best_score_XGB_4 = float('inf');
#    best_para_XGB_4 = {};
#    for ss in [i/10.0 for i in range(5, 10)]:
#        for cb in [i/10.0 for i in range(5, 10)]:
#            clf_XGB_4 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=cb, gamma=best_para_XGB_3['gamma'], learning_rate=0.1, max_depth=best_para_XGB_2['max_depth'], min_child_weight=best_para_XGB_2['min_child_weight'], n_estimators=best_para_XGB_1['n_estimators'], 
#                                  reg_alpha=0.5, reg_lambda=0.5, subsample=ss, silent=1, random_state=9))
#            score_clf_XGB_4 = rmsle_cv(clf_XGB_4, 5, train.values, y_train.values.ravel())
#            score = score_clf_XGB_4.mean()
#            print(ss, ' ', cb, ' score:', score)
#            if score < best_score_XGB_4:
#                best_score_XGB_4 = score
#                best_para_XGB_4 = {'subsample': ss, 'colsample_bytree': cb}
#                
#    # 'subsample':0.6, 'colsample_bytree':0.5, score:0.12015979179052268
#    
#    best_score_XGB_5 = float('inf');
#    best_para_XGB_5 = {};
#    for ra in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#        for rl in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#            clf_XGB_5 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=best_para_XGB_4['colsample_bytree'], gamma=best_para_XGB_3['gamma'], learning_rate=0.1, max_depth=best_para_XGB_2['max_depth'], min_child_weight=best_para_XGB_2['min_child_weight'], n_estimators=best_para_XGB_1['n_estimators'], 
#                                  reg_alpha=ra, reg_lambda=rl, subsample=best_para_XGB_4['subsample'], silent=1, random_state=9))
#            score_clf_XGB_5 = rmsle_cv(clf_XGB_5, 5, train.values, y_train.values.ravel())
#            score = score_clf_XGB_5.mean()
#            print(ra, ' ', rl, ' score:', score)
#            if score < best_score_XGB_5:
#                best_score_XGB_5 = score
#                best_para_XGB_5 = {'reg_alpha': ra, 'reg_lambda': rl}
#                
#    # 'reg_alpha':0.001, 'reg_lambda':1, score:0.11832497642250632
#    
#    clf_XGB = make_pipeline(RobustScaler(), xgb.XGBRegressor(**best_para_XGB_1, **best_para_XGB_2, **best_para_XGB_3, **best_para_XGB_4, **best_para_XGB_5, 
#                            learning_rate=0.1, silent=1, random_state=9))
#    best_score_XGB = best_score_XGB_5
    clf_XGB.fit(train.values, y_train.values.ravel())
    clf_XGB_train_pred = clf_XGB.predict(train.values)
    clf_XGB_train_score = rmsle_all(y_train.values.ravel(), clf_XGB_train_pred)
    clf_XGB_pred = np.expm1(clf_XGB.predict(test.values))
    joblib.dump(clf_XGB, "train_model_XGB.m")
    
    #----------------------------------LGB-------------------------------------
    print('LGB...')
    # 'n_estimators':200, 'max_depth':3, 'num_leaves':5, 'max_bin':95, 'min_data_in_leaf':5, 'feature_fraction':0.5, 'bagging_fraction':0.7, 'bagging_freq':10, 'lambda_l1':0.5, 'lambda_l2':1, score:0.119115960382797, train score:0.0909658162722463
    clf_LGB = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=3, num_leaves=5, learning_rate=0.1, n_estimators=200, max_bin=70, bagging_fraction=0.8, bagging_freq=5, 
                            feature_fraction=0.5, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=5, lambda_l1=0.005, lambda_l2=10))
    best_score_LGB = 0.119115960382797
#    best_score_LGB_1 = float('inf');
#    best_para_LGB_1 = {};
#    for ne in range(100, 1100, 100):
#        clf_LGB_1 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=4, num_leaves=10, learning_rate=0.1, n_estimators=ne, max_bin=50, bagging_fraction=0.8, bagging_freq=5, 
#                                  feature_fraction=0.8, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=10, lambda_l1=0.5, lambda_l2=0.5))
#        score_clf_LGB_1 = rmsle_cv(clf_LGB_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_LGB_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_LGB_1:
#            best_score_LGB_1 = score
#            best_para_LGB_1 = {'n_estimators': ne}
#    
#    # 'n_estimators':200, score:0.12369471248001158
#    
#    best_score_LGB_2 = float('inf');
#    best_para_LGB_2 = {};
#    for md in range(3, 6):
#        for nl in range(md, 2**md):
#            clf_LGB_2 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=md, num_leaves=nl, learning_rate=0.1, n_estimators=best_para_LGB_1['n_estimators'], max_bin=50, bagging_fraction=0.8, bagging_freq=5, 
#                                      feature_fraction=0.8, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=10, lambda_l1=0.5, lambda_l2=0.5))
#            score_clf_LGB_2 = rmsle_cv(clf_LGB_2, 5, train.values, y_train.values.ravel())
#            score = score_clf_LGB_2.mean()
#            print(md, ' ', nl, ' score:', score)
#            if score < best_score_LGB_2:
#                best_score_LGB_2 = score
#                best_para_LGB_2 = {'max_depth': md, 'num_leaves':nl}
#                
#    # 'max_depth':3, 'num_leaves':5, score:0.12208503445224332
#    
#    best_score_LGB_3 = float('inf');
#    best_para_LGB_3 = {};
#    for mb in range(5, 100, 5):
#        for mdil in range(5, 55, 5):
#            clf_LGB_3 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=best_para_LGB_2['max_depth'], num_leaves=best_para_LGB_2['num_leaves'], learning_rate=0.1, n_estimators=best_para_LGB_1['n_estimators'], max_bin=mb, bagging_fraction=0.8, bagging_freq=5, 
#                                      feature_fraction=0.8, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=mdil, lambda_l1=0.5, lambda_l2=0.5))
#            score_clf_LGB_3 = rmsle_cv(clf_LGB_3, 5, train.values, y_train.values.ravel())
#            score = score_clf_LGB_3.mean()
#            print(mb, ' ', mdil, ' score:', score)
#            if score < best_score_LGB_3:
#                best_score_LGB_3 = score
#                best_para_LGB_3 = {'max_bin': mb, 'min_data_in_leaf':mdil}
#    
#    # 'max_bin':95, 'min_data_in_leaf':5, score:0.11997429370758113
#    
#    best_score_LGB_4 = float('inf');
#    best_para_LGB_4 = {};
#    for ff in [i/10.0 for i in range(5, 10)]:
#        for bfr in [i/10.0 for i in range(5, 10)]:
#            for bfre in range(0, 55, 5):
#                clf_LGB_4 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=best_para_LGB_2['max_depth'], num_leaves=best_para_LGB_2['num_leaves'], learning_rate=0.1, n_estimators=best_para_LGB_1['n_estimators'], max_bin=best_para_LGB_3['max_bin'], bagging_fraction=bfr, bagging_freq=bfre, 
#                                          feature_fraction=ff, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=best_para_LGB_3['min_data_in_leaf'], lambda_l1=0.5, lambda_l2=0.5))
#                score_clf_LGB_4 = rmsle_cv(clf_LGB_4, 5, train.values, y_train.values.ravel())
#                score = score_clf_LGB_4.mean()
#                print(ff, ' ', bfr, ' ', bfre, ' score:', score)
#                if score < best_score_LGB_4:
#                    best_score_LGB_4 = score
#                    best_para_LGB_4 = {'feature_fraction': ff, 'bagging_fraction':bfr, 'bagging_freq':bfre}
#                    
#    # 'feature_fraction':0.5, 'bagging_fraction':0.7, 'bagging_freq':10, score:0.11940541566426943
#    
#    best_score_LGB_5 = float('inf');
#    best_para_LGB_5 = {};
#    for ll1 in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#        for ll2 in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#            clf_LGB_5 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=best_para_LGB_2['max_depth'], num_leaves=best_para_LGB_2['num_leaves'], learning_rate=0.1, n_estimators=best_para_LGB_1['n_estimators'], max_bin=best_para_LGB_3['max_bin'], bagging_fraction=best_para_LGB_4['bagging_fraction'], bagging_freq=best_para_LGB_4['bagging_freq'], 
#                                      feature_fraction=best_para_LGB_4['feature_fraction'], feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=best_para_LGB_3['min_data_in_leaf'], lambda_l1=ll1, lambda_l2=ll2))
#            score_clf_LGB_5 = rmsle_cv(clf_LGB_5, 5, train.values, y_train.values.ravel())
#            score = score_clf_LGB_5.mean()
#            print(ll1, ' ', ll2, ' score:', score)
#            if score < best_score_LGB_5:
#                best_score_LGB_5 = score
#                best_para_LGB_5 = {'lambda_l1': ll1, 'lambda_l2':ll2}
#    
#    # 'lambda_l1':0.5, 'lambda_l2':1, score:0.119115960382797
#            
#    clf_LGB = make_pipeline(RobustScaler(), lgb.LGBMRegressor(**best_para_LGB_1, **best_para_LGB_2, **best_para_LGB_3, **best_para_LGB_4, **best_para_LGB_5, 
#                            objective='regression', learning_rate=0.1, feature_fraction_seed=11, bagging_seed=11))
#    best_score_LGB = best_score_LGB_5
    clf_LGB.fit(train.values, y_train.values.ravel())
    clf_LGB_train_pred = clf_LGB.predict(train.values)
    clf_LGB_train_score = rmsle_all(y_train.values.ravel(), clf_LGB_train_pred)
    clf_LGB_pred = np.expm1(clf_LGB.predict(test.values))
    joblib.dump(clf_LGB, "train_model_LGB.m")
    
    #-------------------------------集成融合------------------------------------
    # 1.Stacking
    print('Stacking...')
    # 1.1
    # base model: ['Lasso', 'ENet', 'GBR', ‘RFR’, 'XGB', 'LGB'], meta model: ['SVR'(C=1, gamma=0.1)], score: 0.11262637486925065, train score: 0.08832624844227878, public score: 0.11799
    print('1...')
    best_score_StackingModel1 = 0.11262637486925065
    Stacking_baseModels1 = [clf_Lasso, clf_ENet, clf_GBR, clf_RFR, clf_XGB, clf_LGB]
    clf_SVR_ = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=1, gamma=0.1))
#    best_score_StackingModel1 = float('inf');
#    best_para_StackingModel1_SVR = {};
#    for c in [0.01, 0.5, 0.1, 1, 5, 10, 100]:
#        for g in np.logspace(-2, 2, 5):
#            clf_SVR_ = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=c, gamma=g))
#            Stacking_metaModel1 = clf_SVR_
#            clf_SM = StackingModel(Stacking_baseModels1, Stacking_metaModel1)
#            score_clf_SM = rmsle_cv(clf_SM, 5, train.values, y_train.values.ravel())
#            score = score_clf_SM.mean()
#            print(c, ' ', g, ' score:', score)
#            if score < best_score_StackingModel1:
#                best_score_StackingModel1 = score
#                best_para_StackingModel1_SVR = {'C': c, 'gamma': g}
#    
#    clf_SVR_ = make_pipeline(RobustScaler(), SVR(**best_para_StackingModel1_SVR, kernel='rbf'))
    Stacking_metaModel1 = clf_SVR_
    clf_StackingModel1 = StackingModel(Stacking_baseModels1, Stacking_metaModel1)
    clf_StackingModel1.fit(train.values, y_train.values.ravel())
    clf_StackingModel1_train_pred = clf_StackingModel1.predict(train.values)
    clf_StackingModel1_train_score = rmsle_all(y_train.values.ravel(), clf_StackingModel1_train_pred)
    clf_StackingModel1_pred = np.expm1(clf_StackingModel1.predict(test.values))
    
#    joblib.dump(clf_StackingModel1, "train_model_Stacking1.m")
    
    # 2.Averaging
#    print('Averaging...')
#    # 2.1
#    # 'model': ; 'weights': ; train score: , public score: 
#    averaging1_train_predictions = np.column_stack([])
#    averaging1_predictions = np.column_stack([])
#    predictions1_weights = []
#    averaging1_train_weighted_score = rmsle_all(y_train.values.ravel(), np.average(averaging1_train_predictions, axis=1, weights=predictions1_weights))
#    averaging1_weighted_pred = np.average(averaging1_predictions, axis=1, weights=predictions1_weights)
    
    #-------------------------------提交结果------------------------------------
    print('Submission...')
    submission = pd.DataFrame()
    submission['Id'] = old_test['Id']
    submission['SalePrice'] = clf_StackingModel1_pred
    submission.to_csv('submission.csv', index=False)