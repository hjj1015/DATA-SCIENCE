# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:52:31 2017

@author: Administrator
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


boston = load_boston()
x = boston['data']
y = boston['target']

#数据标准化min-max
min_max = preprocessing.MinMaxScaler().fit(x)
x_minmax = min_max.transform(x)
'''
为了说明正则化防止模型过拟合的效果和约束特征的好处，我们这里需要产生更多的特征
使用PolynomialFeatures函数，产生由多项式形成的新特征，共有104个特征:
104 = 13 + 13 + 13*12/2
使用多项式法产生特征，原有的特征x1...x13共13个；
平方的特征x1^2,x2^2,....x13^2共13个；
还有任意选择两个特征乘积形成的新特征xi*xj，共13*12/2=78个
'''
x_poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_minmax)

x_train,x_test,y_train,y_test=train_test_split(x_poly,y,random_state=0)
lr = LinearRegression().fit(x_train,y_train)

print("Training set score:{:.2f}".format(lr.score(x_train,y_train)))
print("Test set score:{:.2f}".format(lr.score(x_test,y_test)))
lr.coef_

#岭回归，L2惩罚项
from sklearn.linear_model import Ridge

ridge = Ridge().fit(x_train,y_train)
print("Training set score:{:.2f}".format(ridge.score(x_train,y_train)))
print("Test set score:{:.2f}".format(ridge.score(x_test,y_test)))

ridge01 = Ridge(0.1).fit(x_train,y_train)
print("Training set score:{:.2f}".format(ridge01.score(x_train,y_train)))
print("Test set score:{:.2f}".format(ridge01.score(x_test,y_test)))

#lasso回归，L1惩罚项
from sklearn.linear_model import Lasso
lasso =Lasso(alpha=0.01,max_iter=100000).fit(x_train,y_train)
print("Training set score:{:.2f}".format(lasso.score(x_train,y_train)))
print("Test set score:{:.2f}".format(lasso.score(x_test,y_test)))
print("number of features used:{}".format(np.sum(lasso.coef_!=0)))
lasso.coef_


