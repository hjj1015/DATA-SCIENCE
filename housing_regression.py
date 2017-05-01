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


#岭回归，L2惩罚项
from sklearn.linear_model import Ridge

#alpha默认取1,惩罚系数alpha越大，模型越不易过拟合
ridge = Ridge(alpha=1).fit(x_train,y_train)
print("Training set score:{:.2f}".format(ridge.score(x_train,y_train)))
print("Test set score:{:.2f}".format(ridge.score(x_test,y_test)))

ridge10 = Ridge(alpha=10).fit(x_train,y_train)
print("Training set score:{:.2f}".format(ridge10.score(x_train,y_train)))
print("Test set score:{:.2f}".format(ridge10.score(x_test,y_test)))

ridge01 = Ridge(0.1).fit(x_train,y_train)
print("Training set score:{:.2f}".format(ridge01.score(x_train,y_train)))
print("Test set score:{:.2f}".format(ridge01.score(x_test,y_test)))

#作回归系数对应的图形
plt.plot(ridge.coef_,'s',label="Ridge alpha=1")
plt.plot(ridge10.coef_,'^',label="Ridge alpha=10")
plt.plot(ridge01.coef_,'v',label="Ridge alpha=0.1")

plt.plot(lr.coef_,'o',label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
#作一条y=0的水平线
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()

#画学习曲线
def plot_learning_curve(est, x, y):
    from sklearn.model_selection import learning_curve,KFold
    training_set_size, train_scores, test_scores = learning_curve(
        est, x, y, train_sizes=np.linspace(.1, 1, 20), cv=KFold(20, shuffle=True, random_state=1))
    estimator_name = est.__class__.__name__
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--',
                    label="training " + estimator_name)
    plt.plot(training_set_size, test_scores.mean(axis=1), '-',
             label="test " + estimator_name, c=line[0].get_color())
    plt.xlabel('Training set size')
    plt.ylabel('Score (R^2)')
    plt.ylim(0, 1.1)
    
plot_learning_curve(Ridge(alpha=1), x_poly, y)
plot_learning_curve(LinearRegression(), x_poly, y)
plt.legend(loc=(0, 1.05), ncol=2, fontsize=11)


#lasso回归，L1惩罚项
from sklearn.linear_model import Lasso

lasso =Lasso().fit(x_train,y_train)
print("Training set score:{:.2f}".format(lasso.score(x_train,y_train)))
print("Test set score:{:.2f}".format(lasso.score(x_test,y_test)))
print("number of features used:{}".format(np.sum(lasso.coef_!=0)))



lasso001 =Lasso(alpha=0.01,max_iter=100000).fit(x_train,y_train)
print("Training set score:{:.2f}".format(lasso001.score(x_train,y_train)))
print("Test set score:{:.2f}".format(lasso001.score(x_test,y_test)))
print("number of features used:{}".format(np.sum(lasso001.coef_!=0)))

lasso00001 =Lasso(alpha=0.0001,max_iter=100000).fit(x_train,y_train)
print("Training set score:{:.2f}".format(lasso00001.score(x_train,y_train)))
print("Test set score:{:.2f}".format(lasso00001.score(x_test,y_test)))
print("number of features used:{}".format(np.sum(lasso00001.coef_!=0)))

#画出不同模型回归系数图
plt.plot(lasso.coef_,'s',label="lasso alpha=1")
plt.plot(lasso001.coef_,'^',label="lasso alpha=0.01")
plt.plot(lasso00001.coef_,'v',label="lasso alpha=0.0001")

plt.plot(ridge01.coef_,'o',label="Rdige alpha=0.1")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.ylim(-25,25)
plt.legend(ncol=2,loc=(0,1.05))
