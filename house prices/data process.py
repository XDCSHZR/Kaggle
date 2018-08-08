# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 19:16:22 2018

@author: HZR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew
from subprocess import check_output
from sklearn.preprocessing import StandardScaler
from scipy.special import boxcox1p
from sklearn.preprocessing import LabelEncoder

def ignore_warn(*args, **kwargs):
    pass

color = sns.color_palette()
sns.set_style('darkgrid')
warnings.warn = ignore_warn
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv('all/train.csv')
test = pd.read_csv('all/test.csv')

# 删除Id
train_Id = train['Id']
test_Id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# 离群点（不处理，尽量使模型更鲁棒）
def check_outliers(train_temp):
    fig, ax = plt.subplots()
    ax.scatter(x=train_temp['GrLivArea'], y=train_temp['SalePrice'])
#    ax.scatter(x=train_temp['LotArea'], y=train_temp['SalePrice'])
    plt.ylabel('SalePirce', fontsize=13)
    plt.xlabel('GrLivArea', fontsize=13)
#    plt.xlabel('LotArea', fontsize=13)
    plt.show()

check_outliers(train)
newtrain1 = train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<300000)].index)
check_outliers(newtrain1)


# 目标变量
def check_dist(train_temp):
    sns.distplot(train_temp['SalePrice'], fit=norm)
    (mu, sigma) = norm.fit(train_temp['SalePrice'])
    plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePirce distribution')
    fig = plt.figure()
    res = stats.probplot(train_temp['SalePrice'], plot=plt)
    plt.show()

check_dist(train)
# 取对数使目标变量满足正态分布，原目标变量存在偏度
train['SalePrice'] = np.log1p(train['SalePrice'])
# 标准正态
#x = StandardScaler()
#newtrain2['SalePrice'] = x.fit_transform(newtrain2['SalePrice'].reshape(-1, 1))
check_dist(train)

# 特征工程----------------------------------------------------------------------
# 训练数据（含“SalePirce”）的特征相关系数（数值型）
corrmat = train.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat, vmax=0.9, square=True)
# 合并数据集
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop('SalePrice', axis=1, inplace=True)
shape_all_data = all_data.shape


# 处理缺失值
def check_miss(data_temp):
    data_missPercent = data_temp.isnull().sum() / shape_all_data[0] * 100
    data_missPercent = data_missPercent.drop(data_missPercent[data_missPercent==0].index).sort_values(ascending=False)
    miss_data = pd.DataFrame({'Miss Percent': data_missPercent})
    return miss_data

miss_data = check_miss(all_data)
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=miss_data.index, y=miss_data['Miss Percent'])
plt.xlabel('Feature', fontsize=15)
plt.ylabel('Miss Percent', fontsize=15)
plt.title('Miss Percent by Feature', fontsize=15)
# 缺失值填充
# “泳池”：“None”
all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
# “杂项功能”：“None”
all_data['MiscFeature'] = all_data['PoolQC'].fillna('None')
# “小路属性”：“None”
all_data['Alley'] = all_data['Alley'].fillna('None')
# “栅栏”：“None”
all_data['Fence'] = all_data['Fence'].fillna('None')
# “壁炉”：“None”
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
# “道路数量”：按街区划分，“中位数”
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# “车库位置”、“车库是否完工”、“车库质量”、“车库状况”：“None”
for para in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[para] = all_data[para].fillna('None')
# "车库年份"、“车库面积”、“车库车容量”：0
for para in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[para] = all_data[para].fillna(0)
# “地下室类型1完工等级”、“地下室类型2完工等级”、“地下室高度”、“地下室情况”、“地下室曝光”：“None”
for para in ('BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtExposure'):
    all_data[para] = all_data[para].fillna('None')
# “地下室类型1完工面积”、“地下室类型2完工面积”、“地下室未完工面积”、“地下室总面积”、“地下室齐全浴室”、“地下室半浴室”
for para in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[para] = all_data[para].fillna(0)
# “砌体贴面类型”：“None”
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
# “砌体贴面面积”：0
all_data['MasVnrArea'] = all_data['MasVnrType'].fillna(0)
# "区域分类"：“众数”
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
# “可用资源类型”：“众数”
all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0])
# “家庭功能”：“众数”
all_data["Functional"] = all_data["Functional"].fillna(all_data['Functional'].mode()[0])
# “电气系统”：“众数”
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# “房屋外墙类型1”、“房屋外墙类型2”：“众数”
for para in ('Exterior1st', 'Exterior2nd'):
    all_data[para] = all_data[para].fillna(all_data[para].mode()[0])
# “厨房质量”：“众数”
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
# “销售类型”：“众数”
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
miss_data_after = check_miss(all_data)

# Label Encoding，便于处理偏度，在顺序上可能有信息
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# 添加新特征：“总面积”=“地下室”+“一层”+“二层”
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# 数据偏度处理
def check_skew(data_temp):
    numeric_feats = data_temp.dtypes[data_temp.dtypes!='object'].index
    skewed_feats = data_temp[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_feats})
    return skewness

skewness = check_skew(all_data)
skewness = skewness[abs(skewness['Skew'])>0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
skewness_after = check_skew(all_data)

# 哑编码
new_all_data = pd.get_dummies(all_data)

# 存储处理后的全部数据
new_train = new_all_data[:train.shape[0]]
train_label = train['SalePrice']
y_train = pd.DataFrame({'SalePrice': train_label})
new_test = new_all_data[train.shape[0]:]
new_train.to_csv('train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
new_test.to_csv('test.csv', index=False)































