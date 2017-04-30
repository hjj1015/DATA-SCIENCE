# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_recall_curve,average_precision_score,roc_curve,auc
#UCI的汽车评估数据集，独热编码实现'
from sklearn.preprocessing import label_binarize

'''数据是car.txt，其中列名已添加'''
df=pd.read_csv('car.txt')
'''如果没有列名，需要使用下面命令添加列名，
并且上面改为df=pd.read_csv('car.txt'，header=None)
df.columns=['buying','maint','doors','persons','lug_boot','safety','acceptance']
'''
df.head()
'''观测目标分类数据集分布情况，发现并不均匀'''
df["acceptance"].value_counts()
'''
因为自变量均为离散变量，模型训练时要进行处理，首先不使用one-hot编码
只直接转换为数值型编码
'''
for col in df.columns:
    j = 1
    listM = list(set(df[col]))
    for i in listM:
        #对于每列中等于listM的元素,就让它=j,j从数字1开始转换
        df[col][df[col]==i]=j  
        j += 1
df.head()
'''因为样本已经随机排列，不用随机打散，取前80%数据为训练集（1382个）'''
'''这里需要将标签列数值转化为字符串str'''
df["acceptance"] = df["acceptance"].astype(str)
df_train = df.iloc[0:1382,]       
df_test = df.iloc[1382:1728,]
'''使用sklearn.svm包中的SVC构建模型，参数设置核函数kernel和约束惩罚C'''
#这里选用默认的径向基核函数rbf,C采用默认1
car_evaluation_model = SVC(C = 1,kernel = "rbf")
car_evaluation_model.fit(df_train.iloc[:,0:6],df_train["acceptance"])
df_pred = car_evaluation_model.predict(df_test.iloc[:,0:6])
#输出混淆矩阵
print(classification_report(list(df_test["acceptance"]),list(df_pred)))
print(pd.DataFrame(confusion_matrix(list(df_test["acceptance"]),list(df_pred)),\
                   columns = df["acceptance"].value_counts().sort_index().index,\
                    index = df["acceptance"].value_counts().sort_index().index))
#输出预测结果
agreement = df_test["acceptance"] == df_pred
print(agreement.value_counts())
print("Accuracy:",accuracy_score(list(df_test["acceptance"]),list(df_pred)))



'''下面是使用one-hot编码的预测分类效果,重新加载数据'''

df=pd.read_csv('car.txt')
df_acceptance = df["acceptance"]
#one-hot编码，
df = pd.get_dummies(df.iloc[:,0:6],prefix = df.iloc[:,0:6].columns)
'''需要编码后再水平拼接，使用pd.concat()'''
df = pd.concat([df,df_acceptance],axis=1)
'''对分类标签进行处理，之前的转换是按照字母顺序编码的，这里指定'''
df["acceptance"][df.acceptance == 'unacc'] = 4
df["acceptance"][df.acceptance == 'acc'] = 3
df["acceptance"][df.acceptance == 'good'] = 2
df["acceptance"][df.acceptance == 'vgood'] = 1
df_train = df.iloc[0:1500,]
df_test = df.iloc[1500:1728,]
car_evaluation_model = SVC(C = 1,kernel = "rbf")
car_evaluation_model.fit(df_train.iloc[:,0:21],df_train["acceptance"].astype(int))

'''对模型性能进行评估'''
df_pred = car_evaluation_model.predict(df_test.iloc[:,0:21])
print(classification_report(list(df_test["acceptance"]),list(df_pred)))
print(pd.DataFrame(confusion_matrix(list(df_test["acceptance"]),list(df_pred)),\
                   columns = df["acceptance"].value_counts().sort_index().index,\
                    index = df["acceptance"].value_counts().sort_index().index))
agreement = df_test["acceptance"] == df_pred
print(agreement.value_counts())
print("Accuracy:",accuracy_score(list(df_test["acceptance"]),list(df_pred)))




