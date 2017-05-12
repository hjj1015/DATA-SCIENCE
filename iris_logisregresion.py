# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:09:19 2017
共150个样本
@author: Administrator
"""
import pandas as pd

iris = pd.read_csv("iris.txt",sep = ", ")
iris.head()
#观察最后类别的分布情况，是否均衡
iris["Class"].value_counts()
'''
Out[75]: 
 Iris-setosa        50
 Iris-virginica     50
 Iris-versicolor    50
Name:  Class, dtype: int64
'''
#使用seaborn画出点对图来观察鸢尾属花各特征的分布
import seaborn as sns
sns.pairplot(iris,hue="Class")
sns.plt.savefig("irisDist.png")

#对Class字符串转成数值,索引列带空格，是因为数据中有空格，没消除
Class_label_dict = {"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}
iris["Class"] = iris["Class"].map(Class_label_dict)   
'''
也可以使用iris = iris.replace("Iris-setosa",0)  ...
'''
#以类别分组观察各个样本分布
iris.groupby("Class").count()

'''
将数据集切分成训练和测试，定义一个split函数，
以前是breast_cancer_knn是直接调用sklearn包下preprocessing中的
cross_validation.train_test_split函数
'''
import random
def split(iris,test_ratio):
    col_list = list(iris.columns)
    y_col = "Class"
    col_list.remove(y_col)
    rows = []
    for i in set(iris[y_col]):
        rows.extend(random.sample(list(iris[iris[y_col]==i].index),
                                  int(test_ratio*len(iris[iris[y_col]==i]))))
    test_iris = iris.ix[rows]
    train_iris = iris.drop(rows)
    train_iris[col_list].to_csv('train_x.csv',encoding = 'utf-8',index=False)
    test_iris[col_list].to_csv('test_x.csv',encoding='utf-8',index=False)
    train_iris[[y_col]].to_csv('train_y.csv',encoding = 'utf-8',index=False)
    test_iris[[y_col]].to_csv('test_y.csv',encoding = 'utf-8',index=False)
split(iris,0.3)    
'''将切分好的数据读取'''
iris_trainx = pd.read_csv("train_x.csv")
iris_trainy = pd.read_csv("train_y.csv")
iris_testx = pd.read_csv("test_x.csv")
iris_testy = pd.read_csv("test_y.csv")

'''使用训练集构建分类模型'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C = 1e3,solver = 'lbfgs')
'''C是罚项系数的倒数，越大表示对系数惩罚的越小'''
classifier.fit(iris_trainx,iris_trainy["Class"].ravel())
'''
LogisticRegression(C=1000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
'''
'''模型性能评估'''
from sklearn import metrics
iris_pred = classifier.predict(iris_testx)
print(metrics.classification_report(iris_testy,iris_pred))
'''
precision    recall  f1-score   support

          0       1.00      1.00      1.00        15
          1       0.88      0.93      0.90        15
          2       0.93      0.87      0.90        15

avg / total       0.93      0.93      0.93        45

这是一个多分类问题，经典的ROC曲线及其对应的AUC指标并不适用
下面使用混淆矩阵观察预测分类和实际分类情况
'''
# import seaborn as sns 已经导入就不必再导入了
colorMetrics = metrics.confusion_matrix(iris_testy,iris_pred)
'''
array([[15,  0,  0],
       [ 0, 14,  1],
       [ 0,  2, 13]])
'''
#画出混淆矩阵的热点图
sns.heatmap(colorMetrics,annot = True,fmt = 'd')
#计算模型测试集中的正确率
print("Accuracy",metrics.accuracy_score(iris_testy,iris_pred))
'''
Accuracy 0.933333333333
在45个测试样本中，正确预测的有42个，整体正确率为0.93
'''
'''
模型性能提升:
上述实验中，为防止模型过拟合，进行L2正则化，取C=1e3，
经过多次调参，发现C还是有些小以至于正则项惩罚的过重出现欠拟合，尝试后取C=1e5
'''
classifier = LogisticRegression(C = 1e5,solver = 'lbfgs')
'''C是罚项系数的倒数，越大表示对系数惩罚的越小'''
classifier.fit(iris_trainx,iris_trainy["Class"].ravel())
'''
LogisticRegression(C=100000.0, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)
'''
iris_pred = classifier.predict(iris_testx)
print(metrics.classification_report(iris_testy,iris_pred))
'''
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        15
          1       0.93      0.93      0.93        15
          2       0.93      0.93      0.93        15

avg / total       0.96      0.96      0.96        45
'''
colorMetrics = metrics.confusion_matrix(iris_testy,iris_pred)

'''
array([[15,  0,  0],
       [ 0, 14,  1],
       [ 0,  1, 14]])
'''
sns.heatmap(colorMetrics,annot = True,fmt = 'd')

print("Accuracy",metrics.accuracy_score(iris_testy,iris_pred))
'''
Out[26]: 0.9555555555555556
'''

 