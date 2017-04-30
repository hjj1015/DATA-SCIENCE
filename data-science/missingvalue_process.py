# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:55:07 2017

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import cross_val_score
#导入数据
mpg = pd.read_csv('auto-mpg.data',header = None)
mpg.head(7)
#数据结构完整化
mpg_list = ['mpg','cylinders','displacement','horsepower','weight',
            'acceleration','model_year','origin','car name']
n = mpg.index[-1]
result = pd.DataFrame(columns=list(mpg_list))
for i in range(n+1):
    temp = mpg.ix[i][0]
    #字符串操作函数','.join，用于字符串的连接
    alldata = ','.join(temp.split('\t')).split(',') 
    first = ','.join(alldata[0].split()).split(',')
    last = alldata[1].replace('"','')
    first.append(last)
    result.loc[i] = first
result.head(7)
#原来的数据集存在6个以？标识的缺失值，我们用NaN替代，并将拥有缺失值的样本列表如下
#缺失值标签替代
n,m = result.shape
miss_row = []
for i in range(n):
    for j in range(m):
        if result.ix[i][j] == '?':
            miss_row.append(i)
            result.ix[i][j] = np.nan
result.iloc[miss_row]
'''将？号替换为nan，也可以如下操作
---------------------------------------------
首先取出result的所有的值，aa = result.values.ravel()
通过len(aa[aa=='?'])可以知道确实有6个是以？缺失的
---------------------------------------------
使用replace函数将所有？替换为nan
result.iloc[:,:].replace('?',np.nan,inplace=True)
但是这种方法没找出缺失的？所在的行数是哪些，上面的方法打印出了缺失行

'''

#通过pandas的describe方法，可看出每一列的分布，horsepower变量只有391个变量，6个缺失值，所有的数据都是字符型
result.describe()
type(result["horsepower"][1])
#删除无关变量car name
mpg_full = result.drop("car name",axis = 1)
#将字符串类型转换为浮点型
mpg_list = ['mpg','cylinders','displacement','horsepower','weight',
            'acceleration','model_year','origin']
for col in mpg_list:
    mpg_full[col] = mpg_full[col].astype(float)
#删除缺失值所在行    
mpg_full.dropna(axis=0,inplace=True)
mpg_full.describe()  
'''人为构造缺失值，比较不同数据集对模型预测的影响'''
rng = np.random.RandomState()
#将数据框转换成array
X_full = np.array(mpg_full.drop("mpg",axis=1))
Y_full = np.array(mpg_full["mpg"])
n_samples = X_full.shape[0]
n_features = X_full.shape[1]
#用不带缺失值的完全集进行预测
estimator = RandomForestRegressor(random_state=0,n_estimators=100)
score =cross_val_score(estimator,X_full,Y_full).mean()
print("完全集预测分数 = %.2f" % score)

#对75%的行，人为构建缺失值
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                            np.ones(n_missing_samples,
                                    dtype=np.bool)))
#用随机种子打乱missing_samples
rng.shuffle(missing_samples)
'''生成n_missing_samples个指定范围内的整数,这里是0到n_features=7'''
missing_features = rng.randint(0,n_features,n_missing_samples)

#在取出缺失值行的情况下进行预测
X_filtered = X_full[~missing_samples,:]
Y_filtered = Y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0,n_estimators=100)
score = cross_val_score(estimator,X_filtered,Y_filtered).mean()
print("去除缺失值预测 = %.2f" % score)
'''
后面是三种填补缺失值的处理方法，详见数据嗨客缺失值处理mpg-data
'''
'''用均值填补缺失值进行预测'''
#开出另外一块内存，并将X_full数据复制给X_missing
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0],missing_features] = 0
Y_missing = Y_full.copy()
estimator = Pipeline([("imputer",Imputer(missing_values=0,
                                        strategy="mean",
                                        axis=0)),
                        ("forest",RandomForestRegressor(random_state=0,
                                                       n_estimators=100))]) 
score = cross_val_score(estimator,X_missing,Y_missing).mean() 
print("均值填补缺失值预测分数 = %.2f" % score)     

'''中位数填补方法：'''

X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
Y_missing = Y_full.copy()
'''利用管道机制，因为参数要反复使用同一个数据集'''
estimator = Pipeline([("imputer",Imputer(missing_values=0,
                                        strategy="median",
                                        axis=0)),
                        ("forest",RandomForestRegressor(random_state=0,
                                                       n_estimators=100))]) 
score = cross_val_score(estimator, X_missing, Y_missing).mean()
print("中位数填补预测分数 = %.2f"  % score)

'''众数填补方法：'''
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
Y_missing = Y_full.copy()
estimator = Pipeline([("imputer",Imputer(missing_values=0,
                                        strategy="most_frequent",
                                        axis=0)),
                        ("forest",RandomForestRegressor(random_state=0,
                                                       n_estimators=100))]) 
score = cross_val_score(estimator, X_missing, Y_missing).mean()
print("众数填补预测分数 = %.2f"  % score)


          
  
    
            