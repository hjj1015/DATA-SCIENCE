# -*- coding: utf-8 -*-
"""
SVM对光学字符分类预测

@author: Administrator
"""

import pandas as pd

letters = pd.read_csv('letter_recognition.txt',sep=',',header=None)
nm_list = ['letter','xbox','ybox','width','height','onpix','xbar','ybar','x2bar','y2bar','xybar','x2ybar','xy2bar','xedge','xedgey','yedge','yedgex']
'''修改数据列名dataframen.columns'''
letters.columns = nm_list
'''使用value_counts函数观察letter列数据的数量分布，再用sort_index排序'''
letters["letter"].value_counts().sort_index()
''' letters.ix[:,0].value_counts().sort_index()效果一样，这样更好点'''
#观察每个自变量的取值范围
letters.iloc[:,1:].describe()
'''观察发现16个自变量取值范围都在0~15之间，因此不需要做标准化操作,
而且数据集作者已经把数据打散，因此也不需要我们随机打散，直接取前14000个样本（70%）做训练集，
后6000个（30%）样本做为测试集'''
letters_train = letters.iloc[0:14000,]
'''这里如果用letters.ix[0:14000,]，就取了前14001行样本了'''
letters_test = letters.iloc[14000:20000,]

'''
模型训练：用sklearn.svm包中的相关类实现支持向量机算法
有三个类可实现：SVC，NuSVC,LinearSVC
SVC和NuSVC接受的参数有细微的差别，且底层的数学形式不一样
LinearSVC使用的是简单的线性核函数，实现基于liblinear,对大规模样本训练速度更快
三者介绍详见http://scikit-learn.org/stable/modules/svm.html
'''

''' 
本例我们使用SVC，它主要有两个参数，核函数参数和约束惩罚参数C
核函数常取 "linear":线性核函数
         “poly”:多项式核函数
         “rbf”:径向基核函数
         “sigmoid”:sigmoid核函数
约束惩罚参数C：对超过约束条件的样本的惩罚 ，C越大惩罚越大，决策边界越窄 
这里C取默认值，C=1
'''
from sklearn.svm import SVC
letter_recognition_model = SVC(C = 1,kernel = "linear")   
letter_recognition_model.fit(letters_train.iloc[:,1:],letters_train['letter'])  
''' 
模型性能评估 
使用predict函数，用上面训练好的SVM模型在测试集上预测
使用sklearn.metrics相关函数对模型性能进行评估
'''
from sklearn import metrics
letters_pred = letter_recognition_model.predict(letters_test.iloc[:,1:])
print(metrics.classification_report(letters_test["letter"],letters_pred))
print(pd.DataFrame(metrics.confusion_matrix(letters_test["letter"],letters_pred),\
                   columns = letters["letter"].value_counts().sort_index().index,\
                    index = letters["letter"].value_counts().sort_index().index))

'''计算测试集中的预测正确率'''
agreement = letters_test["letter"] == letters_pred
print(agreement.value_counts())
print("Accuracy:",metrics.accuracy_score(letters_test["letter"],letters_pred))
'''模型性能提升'''
'''通过尝试核函数和惩罚参数，试这两个参数来进一步改善模型预测'''
#核函数的选取
kernels = ["rbf","poly","sigmoid"]
for kernel in kernels:
    letters_model = SVC(C = 1,kernel = kernel)
    letters_model.fit(letters_train.iloc[:,1:],letters_train["letter"])
    letters_pred = letters_model.predict(letters_test.iloc[:,1:])
    print("kernel = ",kernel,",Accuracy:",\
          metrics.accuracy_score(letters_test["letter"],letters_pred))

#惩罚参数C选取   
''' 分别测试C = 0.01，0.1,1，10,100 '''
c_list = [0.01,0.1,1,10,100]
for C in c_list:
    letters_model = SVC(C = C,kernel="rbf")
    letters_model.fit(letters_train.iloc[:,1:],letters_train["letter"])
    letters_pred = letters_model.predict(letters_test.iloc[:,1:])
    print("C = ",C,",Accuracy:",\
          metrics.accuracy_score(letters_test["letter"],letters_pred))


    
