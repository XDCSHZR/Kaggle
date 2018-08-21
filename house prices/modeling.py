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
    
   
# Averaging(flag=0: mean(); flag=1: weighted())
class AveragingModel(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, scores):
        self.models = models
        self.scores = scores
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self
    
    def predict(self, X, flag=1):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1) if flag==0 else np.average(predictions, axis=1, weights=[1/x for x in self.scores])
    
    
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
    train = pd.read_csv('train.csv')
    y_train = pd.read_csv('y_train.csv')
    test = pd.read_csv('test.csv')
    old_test = pd.read_csv('all/test.csv')
    
    # --------------------------------Lasso------------------------------------
    # 网格搜索+交叉验证
    # 'alpha':0.0005, score:0.12372815776444199, train score:0.10585555236803794
    clf_lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    best_score_lasso = 0.12372815776444199
    print('lasso...')
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
    clf_lasso.fit(train.values, y_train.values.ravel())
    clf_lasso_train_pred = clf_lasso.predict(train.values)
    clf_lasso_train_score = rmsle_all(y_train.values.ravel(), clf_lasso_train_pred)
    # “SalePrice”之前经过log(1+x)
    clf_lasso_pred = np.expm1(clf_lasso.predict(test.values))
    # 模型持久化
    joblib.dump(clf_lasso, "train_model_lasso.m")
#    clf_lasso = joblib.load("train_model_lasso.m")
    
    # --------------------------------ENet-------------------------------------
    # 'alpha':0.001, l1_ratio:0.5, score:0.12355203588621444, train score:0.10863538284636313
    clf_ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=3))
    best_score_ENet = 0.12355203588621444
    print('ENet...')
#    best_score_ENet = float('inf');
#    best_para_ENet = {};
#    for a in [0.0005, 0.001, 0.01, 0.1, 1, 10, 100]:
#        for l1r in [0.9, 0.8, 0.7, 0.6, 0.5]:
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
     # 'C':1, 'gamma':0.01, score:0.14908849519032602, train score:0.08747106958493701
    clf_SVR = make_pipeline(RobustScaler(), SVR(kernel='rbf', C=1, gamma=0.01)) 
    best_score_SVR = 0.14908849519032602
    print('SVR...')
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
    # 'n_estimators':400, 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':20, score:0.12474860174235176, train score:0.07614786383988222
    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=5, max_features='sqrt', 
                            min_samples_leaf=20, min_samples_split=50, loss='huber', random_state=5))
    best_score_GBR = 0.12474860174235176
    print('GBR...')
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
#    # 'n_estimators':400, score:0.12511891681995607
#    
#    best_score_GBR_2 = float('inf');
#    best_para_GBR_2 = {};
#    for md in range(3, 6):
#        for mss in range(50, 100, 10):
#            for msl in range(10, 40, 10):
#                clf_GBR_2 = make_pipeline(RobustScaler(), GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=md, max_features='sqrt', 
#                                          min_samples_leaf=msl, min_samples_split=mss, loss='huber', random_state=5))
#                score_clf_GBR_2 = rmsle_cv(clf_GBR_2, 5, train.values, y_train.values.ravel())
#                score = score_clf_GBR_2.mean()
#                print(md, ' ', mss, ' ', msl, ' score:', score)
#                if score < best_score_GBR_2:
#                    best_score_GBR_2 = score
#                    best_para_GBR_2 = {'max_depth': md, 'min_samples_split':mss, 'min_samples_leaf':msl}
#    
#    # 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':20, score:0.12474860174235176
#    
#    clf_GBR = make_pipeline(RobustScaler(), GradientBoostingRegressor(**best_para_GBR_1, **best_para_GBR_2, learning_rate=0.1,  max_features='sqrt', loss='huber', random_state=5))
    best_score_GBR = 0.12474860174235176
    clf_GBR.fit(train.values, y_train.values.ravel())
    clf_GBR_train_pred = clf_GBR.predict(train.values)
    clf_GBR_train_score = rmsle_all(y_train.values.ravel(), clf_GBR_train_pred)
    clf_GBR_pred = np.expm1(clf_GBR.predict(test.values))
    joblib.dump(clf_GBR, "train_model_GBR.m")
    
    # ---------------------------------RFR-------------------------------------
    # 'n_estimators':200, 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':10, score:0.17941187986702664, train score:0.1659306614322656
    clf_RFR = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=200, max_depth=5, max_features='sqrt', 
                            min_samples_leaf=10, min_samples_split=50, random_state=7))
    best_score_RFR = 0.17941187986702664
    print('RFR...')
#    best_score_RFR_1 = float('inf');
#    best_para_RFR_1 = {};
#    for ne in range(100, 1100, 100):
#        clf_RFR_1 = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=ne, max_depth=4, max_features='sqrt', 
#                                  min_samples_leaf=15, min_samples_split=50, random_state=7))
#        score_clf_RFR_1 = rmsle_cv(clf_RFR_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_RFR_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_RFR_1:
#            best_score_RFR_1 = score
#            best_para_RFR_1 = {'n_estimators': ne}
#    
#    # 'n_estimators':200, score:0.18996203705289325
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
#    # 'max_depth':5, 'min_samples_split':50, 'min_samples_leaf':10, score:0.17941187986702664
#    
#    clf_RFR = make_pipeline(RobustScaler(), RandomForestRegressor(**best_para_RFR_1, **best_para_RFR_2, max_features='sqrt', random_state=7))
#    best_score_RFR = 0.17941187986702664
    clf_RFR.fit(train.values, y_train.values.ravel())
    clf_RFR_train_pred = clf_RFR.predict(train.values)
    clf_RFR_train_score = rmsle_all(y_train.values.ravel(), clf_RFR_train_pred)
    clf_RFR_pred = np.expm1(clf_RFR.predict(test.values))
    joblib.dump(clf_RFR, "train_model_RFR.m")
    
    #----------------------------------XGB-------------------------------------
    # 'n_estimators':900, 'max_depth':4, 'min_child_weight':3, 'gamma':0.0, 'subsample':0.8, 'colsample_bytree':0.6, 'reg_alpha':0.005, 'reg_lambda':1, score:0.12080521195456355, train score:0.009121506649007956
    clf_XGB = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.0, learning_rate=0.1, max_depth=4, min_child_weight=3, n_estimators=900, 
                            reg_alpha=0.005, reg_lambda=1, subsample=0.8, silent=1, random_state=9))
    best_score_XGB = 0.12080521195456355
    print('XGB...')
#    best_score_XGB_1 = float('inf');
#    best_para_XGB_1 = {};
#    for ne in range(100, 1100, 100):
#        clf_XGB_1 = make_pipeline(RobustScaler(), xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.1, learning_rate=0.1, max_depth=4, min_child_weight=2, n_estimators=ne, 
#                                  reg_alpha=0.5, reg_lambda=0.5, subsample=0.8, silent=1, random_state=9))
#        score_clf_XGB_1 = rmsle_cv(clf_XGB_1, 5, train.values, y_train.values.ravel())
#        score = score_clf_XGB_1.mean()
#        print(ne, ' score:', score)
#        if score < best_score_XGB_1:
#            best_score_XGB_1 = score
#            best_para_XGB_1 = {'n_estimators': ne}
#            
#    # 'n_estimators':900, score:0.12688135132237224
#    
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
#    # 'max_depth':4, 'min_child_weight':3, score:0.12650157584235383
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
#    # 'gamma':0.0, score:0.12264064189257973
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
#    # 'subsample':0.8, 'colsample_bytree':0.6, score:0.12248465412860639
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
#    # 'reg_alpha':0.005, 'reg_lambda':1, score:0.12080521195456355
#    
#    clf_XGB = make_pipeline(RobustScaler(), xgb.XGBRegressor(**best_para_XGB_1, **best_para_XGB_2, **best_para_XGB_3, **best_para_XGB_4, **best_para_XGB_5, 
#                            learning_rate=0.1, silent=1, random_state=9))
#    best_score_XGB = 0.12080521195456355
    clf_XGB.fit(train.values, y_train.values.ravel())
    clf_XGB_train_pred = clf_XGB.predict(train.values)
    clf_XGB_train_score = rmsle_all(y_train.values.ravel(), clf_XGB_train_pred)
    clf_XGB_pred = np.expm1(clf_XGB.predict(test.values))
    joblib.dump(clf_XGB, "train_model_XGB.m")
    
    #----------------------------------LGB-------------------------------------
   # 'n_estimators':300, 'max_depth':3, 'num_leaves':5, 'max_bin':25, 'min_data_in_leaf':5, 'feature_fraction':0.5, 'bagging_fraction':0.9, 'bagging_freq':20, 'lambda_l1':0.1, 'lambda_l2':0.1, score:0.12246028009192222, train score:0.07824464480662245
    clf_LGB = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=3, num_leaves=5, learning_rate=0.1, n_estimators=300, max_bin=25, bagging_fraction=0.9, bagging_freq=20, 
                            feature_fraction=0.5, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=5, lambda_l1=0.1, lambda_l2=0.1))
    best_score_LGB = 0.12246028009192222
    print('LGB...')
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
#    # 'n_estimators':300, score:0.1288800351018546
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
#    # 'max_depth':3, 'num_leaves':5, score:0.12662570527825173
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
#    # 'max_bin':25, 'min_data_in_leaf':5, score:0.12399329008843996
#    
#    best_score_LGB_4 = float('inf');
#    best_para_LGB_4 = {};
#    for ff in [i/10.0 for i in range(5, 10)]:
#        for bfr in [i/10.0 for i in range(5, 10)]:
#            for bfre in range(0, 55, 5):
#                clf_LGB_4 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=3, num_leaves=5, learning_rate=0.1, n_estimators=300, max_bin=25, bagging_fraction=bfr, bagging_freq=bfre, 
#                                      feature_fraction=ff, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=5, lambda_l1=0.5, lambda_l2=0.5))
#                score_clf_LGB_4 = rmsle_cv(clf_LGB_4, 5, train.values, y_train.values.ravel())
#                score = score_clf_LGB_4.mean()
#                print(ff, ' ', bfr, ' ', bfre, ' score:', score)
#                if score < best_score_LGB_4:
#                    best_score_LGB_4 = score
#                    best_para_LGB_4 = {'feature_fraction': ff, 'bagging_fraction':bfr, 'bagging_freq':bfre}
#                    
#    # 'feature_fraction':0.5, 'bagging_fraction':0.9, 'bagging_freq':20, score:0.12268825212436378
#    
#    best_score_LGB_5 = float('inf');
#    best_para_LGB_5 = {};
#    for ll1 in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#        for ll2 in [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]:
#            clf_LGB_5 = make_pipeline(RobustScaler(), lgb.LGBMRegressor(objective='regression', max_depth=3, num_leaves=5, learning_rate=0.1, n_estimators=300, max_bin=25, bagging_fraction=0.9, bagging_freq=20, 
#                                      feature_fraction=0.5, feature_fraction_seed=11, bagging_seed=11, min_data_in_leaf=5, lambda_l1=ll1, lambda_l2=ll2))
#            score_clf_LGB_5 = rmsle_cv(clf_LGB_5, 5, train.values, y_train.values.ravel())
#            score = score_clf_LGB_5.mean()
#            print(ll1, ' ', ll2, ' score:', score)
#            if score < best_score_LGB_5:
#                best_score_LGB_5 = score
#                best_para_LGB_5 = {'lambda_l1': ll1, 'lambda_l2':ll2}
#    
#    # 'lambda_l1':0.1, 'lambda_l2':0.1, score:0.12246028009192222
#            
#    clf_LGB = make_pipeline(RobustScaler(), lgb.LGBMRegressor(**best_para_LGB_1, **best_para_LGB_2, **best_para_LGB_3, **best_para_LGB_4, **best_para_LGB_5, 
#                            objective='regression', learning_rate=0.1, feature_fraction_seed=11, bagging_seed=11))
#    best_score_LGB = 0.12246028009192222
    clf_LGB.fit(train.values, y_train.values.ravel())
    clf_LGB_train_pred = clf_LGB.predict(train.values)
    clf_LGB_train_score = rmsle_all(y_train.values.ravel(), clf_LGB_train_pred)
    clf_LGB_pred = np.expm1(clf_LGB.predict(test.values))
    joblib.dump(clf_LGB, "train_model_LGB.m")
    
    #-------------------------------集成融合------------------------------------
    # 1.Averaging
#    # model:['lasso', 'ENet', 'SVR', 'GBR'], score:0.12024912525937
##    best_score_fusionModel = 0.12024912525937
#    print('fusionModel...')
#    fusion_baseModels = [clf_lasso, clf_ENet, clf_SVR, clf_GBR]
#    fusion_baseScores = [best_score_lasso, best_score_ENet, best_score_SVR, best_score_GBR]
#    clf_fusionModel = AveragingModel(fusion_baseModels, fusion_baseScores)
#    score_clf_fusionModel = rmsle_cv(clf_fusionModel, 5, train.values, y_train.values.ravel())
#    best_score_fusionModel = score_clf_fusionModel.mean()
#    clf_fusionModel.fit(train.values, y_train.values.ravel())
#    clf_fusionModel_train_pred = clf_fusionModel.predict(train.values)
#    clf_fusionModel_train_score = rmsle_all(y_train.values.ravel(), clf_fusionModel_train_pred)
#    clf_fusionModel_pred = np.expm1(clf_fusionModel.predict(test.values))
#    joblib.dump(clf_fusionModel, "train_model_fusion.m")
    
    # 2.Stacking + Averaging
    # base model:['ENet', 'SVR', 'GBR', 'RFR'], meta model:['lasso'], score:0.12057812824120848, train score:0.0887370671205139
#    best_score_StackingModel = 0.12057812824120848
    print('Stacking...')
    Stacking_baseModels = [clf_ENet, clf_SVR, clf_GBR, clf_RFR]
    Stacking_metaModel = clf_lasso
#    clf_StackingModel = StackingRegressor(regressors=Stacking_baseModels, meta_regressor=Stacking_metaModel)
#    clf_StackingModel.fit(train.values, y_train.values.ravel())
#    clf_StackingModel.predict(test.values)
    clf_StackingModel = StackingModel(Stacking_baseModels, Stacking_metaModel)
    score_clf_StackingModel = rmsle_cv(clf_StackingModel, 5, train.values, y_train.values.ravel())
    best_score_StackingModel = score_clf_StackingModel.mean()
    clf_StackingModel.fit(train.values, y_train.values.ravel())
    clf_StackingModel_train_pred = clf_StackingModel.predict(train.values)
    clf_StackingModel_train_score = rmsle_all(y_train.values.ravel(), clf_StackingModel_train_pred)
    clf_StackingModel_pred = np.expm1(clf_StackingModel.predict(test.values))
    joblib.dump(clf_StackingModel, "train_model_Stacking.m")
    # final averaging, train score(mean, weight1, weight2):0.055196425241981026, 0.05512746294054255, 0.07120163071919022
    final_train_predictions = np.column_stack([clf_StackingModel_train_pred, clf_XGB_train_pred, clf_LGB_train_pred])
    final_predictions = np.column_stack([clf_StackingModel_pred, clf_XGB_pred, clf_LGB_pred])
    predictions_weights_1 = [1/best_score_StackingModel, 1/best_score_XGB, 1/best_score_LGB]
    predictions_weights_2 = [0.6, 0.15, 0.25]
    final_train_mean_score = rmsle_all(y_train.values.ravel(), np.mean(final_train_predictions, axis=1))
    final_train_weighted_score_1 = rmsle_all(y_train.values.ravel(), np.average(final_train_predictions, axis=1, weights=predictions_weights_1))
    final_train_weighted_score_2 = rmsle_all(y_train.values.ravel(), np.average(final_train_predictions, axis=1, weights=predictions_weights_2))
    final_mean_pred = np.mean(final_predictions, axis=1)
    final_weighted_pred_1 = np.average(final_predictions, axis=1, weights=predictions_weights_1)
    final_weighted_pred_2 = np.average(final_predictions, axis=1, weights=predictions_weights_2)
    
    #-------------------------------提交结果------------------------------------
    submission = pd.DataFrame()
    submission['Id'] = old_test['Id']
    submission['SalePrice'] = final_weighted_pred_2
    submission.to_csv('submission.csv', index=False)












































