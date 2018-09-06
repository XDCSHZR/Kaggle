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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 一般参数设置
def ignore_warn(*args, **kwargs):
    pass

def plotfeats(frame, feats, kind, cols=4):
    """批量绘图函数。
    
    Parameters
    ----------
    frame : pandas.DataFrame
        待绘图的数据
    
    feats : list 或 numpy.array
        待绘图的列名称
        
    kind : str
        绘图格式：'hist'-直方图；'scatter'-散点图；'hs'-直方图和散点图隔行交替；'box'-箱线图，每个feat一幅图；'boxp'-Price做纵轴，feat做横轴的箱线图。
        
    cols : int
        每行绘制几幅图
    
    Returns
    -------
    None
    """
    rows = int(np.ceil((len(feats))/cols))
    if rows==1 and len(feats)<cols:
        cols = len(feats)
    #print("输入%d个特征，分%d行、%d列绘图" % (len(feats), rows, cols))
    if kind == 'hs': #hs:hist and scatter
        fig, axes = plt.subplots(nrows=rows*2,ncols=cols,figsize=(cols*5,rows*10))
    else:
        fig, axes = plt.subplots(nrows=rows,ncols=cols,figsize=(cols*5,rows*5))
        if rows==1 and cols==1:
            axes = np.array([axes])
        axes = axes.reshape(rows,cols) # 当 rows=1 时，axes.shape:(cols,)，需要reshape一下
    i=0
    for f in feats:
        #print(int(i/cols),i%cols)
        if kind == 'hist':
            #frame.hist(f,bins=100,ax=axes[int(i/cols),i%cols])
            frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols),i%cols])
        elif kind == 'scatter':
            frame.plot.scatter(x=f,y='SalePrice', ax=axes[int(i/cols),i%cols])
        elif kind == 'hs':
            frame.plot.hist(y=f,bins=100,ax=axes[int(i/cols)*2,i%cols])
            frame.plot.scatter(x=f,y='SalePrice', ax=axes[int(i/cols)*2+1,i%cols])
        elif kind == 'box':
            frame.plot.box(y=f,ax=axes[int(i/cols),i%cols])
        elif kind == 'boxp':
            sns.boxplot(x=f,y='SalePrice', data=frame, ax=axes[int(i/cols),i%cols])
        i += 1
    plt.show()

color = sns.color_palette()
sns.set_style('darkgrid')
warnings.warn = ignore_warn
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# ------------------------------特征工程----------------------------------------
# 读取数据
train = pd.read_csv('all/train.csv')
test = pd.read_csv('all/test.csv')

# 删除Id
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# 按照数据描述，删除面积>4000的离群点，其它延趋势轴的“有可能的离群点”不做处理，尽量使模型更鲁棒
def check_outliers(train_temp, feature):
    fig, ax = plt.subplots()
    ax.scatter(x=train_temp[feature], y=train_temp['SalePrice'])
    plt.xlabel(feature, fontsize=13)
    plt.ylabel('SalePirce', fontsize=13)
    plt.show()

check_outliers(train, 'GrLivArea')
train = train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<300000)].index)
check_outliers(train, 'GrLivArea')

# 目标变量处理，正态纠偏
def check_dist(train_temp):
    sns.distplot(train_temp['SalePrice'], fit=norm)
    (mu, sigma) = norm.fit(train_temp['SalePrice'])
    plt.legend(['Normal dist. ($\mu=${:.2f} and $\sigma=${:.2f})'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePirce distribution')
    fig = plt.figure()
    # QQ图，是否符合正态分布
    res = stats.probplot(train_temp['SalePrice'], plot=plt)
    plt.show()

check_dist(train)
# 取对数使目标变量满足正态分布，原目标变量存在偏度
train['SalePrice'] = np.log1p(train['SalePrice'])
# 标准正态
#x = StandardScaler()
#train['SalePrice'] = x.fit_transform(train['SalePrice'].reshape(-1, 1))
check_dist(train)

# 合并数据集
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop('SalePrice', axis=1, inplace=True)
shape_all_data = all_data.shape

# 缺失值处理 1
# “伪缺失值”处理（原数据描述中的NA，数据中会被默认为缺失）
# “伪缺失值”特征
features_obj_miss = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                     'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                     'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
# 若某条样本没有车库，则该样本（若有缺失）的'GarageYrBlt'：'最小值', 'GarageCars'：0, 'GarageArea'：0
# 若某条样本没有地下室，则该样本（若有缺失）的'BsmtFinSF1':0, 'BsmtFinSF2':0, 'BsmtFullBath':0, 'BsmtHalfBath':0, 'BsmtUnfSF':0, 'TotalBsmtSF':0
feature_Garage = 'GarageType'
feature_Garage_num_miss_sp = 'GarageYrBlt'
feature_Garage_num_miss = ['GarageArea', 'GarageCars']
feature_Bsmt = 'BsmtQual';
feature_Bsmt_num_miss = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
# 处理字符型“伪缺失值”
for e in features_obj_miss:
    train[e] = train[e].fillna('None')
    all_data[e] = all_data[e].fillna('None')

# 处理部分相关缺失值
for i in all_data[all_data[feature_Garage]=='None'].index:
    if np.isnan(all_data.loc[i, feature_Garage_num_miss_sp]):
        all_data.loc[i, feature_Garage_num_miss_sp] = all_data[feature_Garage_num_miss_sp].min()
        
for i in train[train[feature_Garage]=='None'].index:
    if np.isnan(train.loc[i, feature_Garage_num_miss_sp]):
        train.loc[i, feature_Garage_num_miss_sp] = all_data[feature_Garage_num_miss_sp].min()

for e in feature_Garage_num_miss:
    for i in all_data[all_data[feature_Garage]=='None'].index:
        if np.isnan(all_data.loc[i, e]):
            all_data.loc[i, e] = 0
    for i in train[train[feature_Garage]=='None'].index:
        if np.isnan(train.loc[i, e]):
            train.loc[i, e] = 0
            
for e in feature_Bsmt_num_miss:
    for i in all_data[all_data[feature_Bsmt]=='None'].index:
        if np.isnan(all_data.loc[i, e]):
            all_data.loc[i, e] = 0
    for i in train[train[feature_Bsmt]=='None'].index:
        if np.isnan(train.loc[i, e]):
            train.loc[i, e] = 0

# 观察各特征分布
# 数值型(连续型，离散型)， 字符型（离散型）
features_num = all_data.dtypes[all_data.dtypes!='object'].index.values
features_obj = all_data.dtypes[all_data.dtypes=='object'].index.values
features_num_discrete = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 
                         'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
                         'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
                         'MoSold', 'YrSold']
features_continue = features_num.copy()
features_discrete = features_obj.copy()
for f in features_num_discrete:
    features_continue = np.delete(features_continue, np.where(features_continue==f))
    features_discrete = np.append(features_discrete, f)

# 散点图（数值型）
# '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'GarageArea', 'GrLivArea', 'LotArea', 'LotFrontage', 'TotalBsmtSF'与'SalePrice'具有较强的正相关趋势
plotfeats(train, features_continue, 'scatter', cols=5)
# 'OverallQual', 'OverallCond', 'FullBath', 'TotRmsAbvGrd', 'Fireplace', 'GarageCars', 与'SalePrice'具有较强的正相关趋势, 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'与'SalePrice'具有一定的正相关趋势
plotfeats(train, features_num_discrete, 'scatter', cols=5)

# 箱线图（加强版散点图：平均水平（中位数），主范围区域（波动程度），异常值）
plotfeats(train, features_continue, kind='box', cols=5)
plotfeats(train, features_num_discrete, kind='boxp', cols=5)
plotfeats(train, features_obj, kind='boxp', cols=5)

# 方差分析
# 协方差，相关系数（皮尔逊相关系数，含'SalePirce'的训练数据，数值型）
corr_pearson = train.corr(method='pearson')
# 'TotalBsmtSF', '1stFlrSF'; 'GarageCars', 'GarageArea'; 'TotRmsAbvGrd', 'GrLivArea', 相关性较高（多重共线性）
plt.subplots(figsize=(20, 20))
sns.heatmap(corr_pearson, vmax=0.9, square=True)

# 删除分析出的不重要特征
# 散点图，箱线图，'MoSold', 'YrSold'较平均，可删除
features_del1 = ['MoSold', 'YrSold']
# 相关系数热图，'TotalBsmtSF', '1stFlrSF'; 'GarageCars', 'GarageArea'; 'TotRmsAbvGrd', 'GrLivArea', 多重共线性，可各自删除其一
#features_del2 = ['TotalBsmtSF', 'GarageCars', 'TotRmsAbvGrd']
features_del2 = []
all_data.drop(features_del1+features_del2, axis=1, inplace=True)
for f in features_del1:
    features_num = np.delete(features_num, np.where(features_num==f))
    features_obj = np.delete(features_obj, np.where(features_obj==f))
    features_num_discrete = np.delete(features_num_discrete, np.where(features_num_discrete==f))
    features_continue = np.delete(features_continue, np.where(features_continue==f))
    features_discrete = np.delete(features_discrete, np.where(features_discrete==f))

# 根据散点图，删除其它一些较明显的离群点（具有较强正相关趋势的特征）
check_outliers(train, 'LotArea')
index_LotArea = train[train['LotArea']>150000].index
train = train.drop(index_LotArea)
all_data = all_data.drop(index_LotArea)
check_outliers(train, 'LotArea')
check_outliers(train, 'LotFrontage')
index_LotFrontage = train[train['LotFrontage']>300].index
train = train.drop(index_LotFrontage)
all_data = all_data.drop(index_LotFrontage)
check_outliers(train, 'LotFrontage')

# 缺失值处理 2
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
# “道路数量”：按街区划分，“中位数”
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# “砌体贴面类型”：“众数”
all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])
# “砌体贴面面积”：“砌体贴面类型”为"None", 0；否则非0数据的“中位数”
for i in all_data[all_data['MasVnrType']=='None'].index:
    if np.isnan(all_data.loc[i, 'MasVnrArea']):
        all_data.loc[i, 'MasVnrArea'] = 0
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data[all_data['MasVnrArea']!=0].median()['MasVnrArea'])
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
# “车库面积”：非0数据的“中位数”
all_data['GarageArea'] = all_data['GarageArea'].fillna(all_data[all_data['GarageArea']!=0].median()['GarageArea'])
# “车库容量”：非0数据的“中位数”
all_data['GarageCars'] = all_data['GarageCars'].fillna(all_data[all_data['GarageCars']!=0].median()['GarageCars'])
# “车库建造日期”：“中位数”
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['GarageYrBlt'].median())
miss_data_after = check_miss(all_data)

# Label Encoding（对'SalePrice'有顺序信息的特征进行顺序编码）
# 自定义'Label Encoding'，按各特征取值分组，并按'SalePrice'升序排序
def encode(frame, feature, targetfeature='SalePrice'):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, targetfeature]].groupby(feature).mean()[targetfeature]
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(0, ordering.shape[0])
    ordering = ordering['order'].to_dict()
    return ordering

features_le = ('MSZoning', 'Street', 'Alley', 'LotShape', 'Utilities', 
               'LotConfig', 'LandSlope', 'Neighborhood', 'HouseStyle', 'MasVnrType', 
               'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 
               'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 
               'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 
               'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence')
for feat in features_le:
    for attr_v, score in encode(train, feat).items():
        all_data.loc[all_data[feat]==attr_v, feat] = score

# 个别数字型特征进行'One-hot Encoding'
features_le_sp = 'MSSubClass'
le = LabelEncoder()
le.fit(list(all_data[features_le_sp].values))
all_data[features_le_sp] = le.transform(list(all_data[features_le_sp].values))
size_mapping = {}
for i in range(0, len(all_data['MSSubClass'].unique())):
    size_mapping[i] = str(i)
    
all_data[features_le_sp] = all_data[features_le_sp].map(size_mapping)

# 添加新特征：“总面积”=“一层”+“二层”
all_data['TotalSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 对字符型的特征进行哑编码（不含顺序信息）
new_all_data = pd.get_dummies(all_data)

# ------------------------------数据处理结果存储--------------------------------
new_train = new_all_data[:train.shape[0]]
train_label = train['SalePrice']
y_train = pd.DataFrame({'SalePrice': train_label})
new_test = new_all_data[train.shape[0]:]
new_train.to_csv('train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
new_test.to_csv('test.csv', index=False)