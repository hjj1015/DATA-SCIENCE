# -*- coding: utf-8 -*-
"""

@author: Administrator
"""
"""
数据集来源于UCI的breast_cancer,使用knn算法对癌细胞进行分类。
乳腺癌数据包括569例乳腺细胞活检样本，每个样本包含32个变量。diagnosis变量是目标变量
（M代表恶性，B代表良性），id变量是样本识别的ID
其他30个变量都是由10个数字化细胞核的10个不同特征的均值、标准差和最大值构成
radius(半径)，texture（质地），perimeter（周长），area(面积)
smoothness(光滑度)，compactness(致密性=perimeter^2/area -1.0)
concavity(凹度)，concave point（凹点），symmetry(对称性)，
fractal dimension(分形维数)
实际数据列名应是
breast_cancer.columns = ["id","diagnosis","radius_mean","texture_mean",
                         "perimeter_mean","area_mean",'smoothness_mean',
                         'compactness_mean','concavity_mean',...太多了]
"""

import pandas as pd
breast_cancer = pd.read_csv("breast_cancer.txt",header = None)
#第一个变量为ID变量，无法为实际的模型构提供有用的信息，所以需要将其删除
del breast_cancer[0]
'''
diagnosis变量是我们的目标变量，首先统计下它的取值分布，即恶性M和良性B的分布情况。
同时，因为diagnosis变量为字符串格式，建模前需要将其转为整数编码
'''
print(breast_cancer[1].value_counts())
print(breast_cancer[1].value_counts()/len(breast_cancer))
diagnosis_dict = {"B":0,"M":1}
breast_cancer[1] = breast_cancer[1].map(diagnosis_dict)
'''
作为示例，我们详细观察30个自变量中的三个变量：
radius_mean,area_mean和smoothness_mean
也就是现在没有注明列名的默认的2、5、6列名
'''
breast_cancer[[2,5,6]].describe()
# breast_cancer[[“radius_mean”,area_mean,smoothness_mean]].describe()
'''
观察发现这三个变量取值范围差别很大，对于KNN来说，尺度不一致会影响Knn的样本距离计算
因此需要对变量进行：标准化
常见有min-max和Z-score标准，这里用min-max
'''
#定义一个min-max标准化函数
def min_max_normalize(x):
    #这里参数x是一个向量
    return (x - x.min())/(x.max() - x.min())

for col in breast_cancer.columns[1:31]:
    breast_cancer[col] = min_max_normalize(breast_cancer[col])
breast_cancer.iloc[:,1:].describe() #从第2列开始显示

'''
划分训练集和测试集
本例我们用数据的70%（398个）训练模型，30%(171个样本)测试

'''
from sklearn import cross_validation
y = breast_cancer[1]
del breast_cancer[1]
x = breast_cancer
'''使用cross_validation下的train_test_split划分训练集测试集'''
breast_cancer_minmax_train,breast_cancer_minmax_test,\
breast_cancer_train_labels,breast_cancer_test_labels \
= cross_validation.train_test_split(x,y,test_size=0.3,random_state=0)
'''观察训练集和测试集恶性样本和良性样本数量分布是否接近'''
print(breast_cancer_train_labels.value_counts()/len(breast_cancer_train_labels))
print(breast_cancer_test_labels.value_counts()/len(breast_cancer_test_labels))

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors =21) #k近邻数为21
knn_model.fit(breast_cancer_minmax_train,breast_cancer_train_labels)

breast_cancer_test_pred = knn_model.predict(breast_cancer_minmax_test)

''' 模型性能评估'''
'''对比预测值pred和真实标签,使用sklearn.metrics包中的相关函数计算评估结果'''
from sklearn import metrics
print(metrics.classification_report(breast_cancer_test_labels,breast_cancer_test_pred))
print(metrics.confusion_matrix(breast_cancer_test_labels,breast_cancer_test_pred))
print(metrics.accuracy_score(breast_cancer_test_labels,breast_cancer_test_pred))
'''
假阳性的样本判错了7个，共63个恶性肿瘤
假阴性指本来是好的预测成坏的（这个错误影响不严重）
假阳性指本来是坏的预测成好的了（这个错误就很严重了）
'''
'''
模型性能提升:
    方法1：尝试不同的k值
    方法2：选择不同的标准化数据
'''
k_list = [1,5,9,11,15,21,27]
for k in k_list:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(breast_cancer_minmax_train,breast_cancer_train_labels)
    breast_cancer_test_pred = knn_model.predict(breast_cancer_minmax_test)
    accuracy = metrics.accuracy_score(breast_cancer_test_labels,breast_cancer_test_pred)
    confusion_mat = metrics.confusion_matrix(breast_cancer_test_labels,breast_cancer_test_pred)
    print("k = ",k)
    print("\t正确率：",'%.2f'%(accuracy*100) + "%")
    print("\t假阴性：",confusion_mat[0,1])
    print("","\t假阳性：",confusion_mat[1,0])
'''
k =  1
        正确率： 92.98%
        假阴性： 7
        假阳性： 5
k =  5
        正确率： 97.08%
        假阴性： 1
        假阳性： 4
k =  9
        正确率： 96.49%
        假阴性： 1
        假阳性： 5
k =  11
        正确率： 95.91%
        假阴性： 2
        假阳性： 5
k =  15
        正确率： 96.49%
        假阴性： 1
        假阳性： 5
k =  21
        正确率： 94.74%
        假阴性： 2
        假阳性： 7
k =  27
        正确率： 94.74%
        假阴性： 2
        假阳性： 7
可见：在k=5时，假阳性数量最少，且假阴性也只有1，正确率达到最高
'''
#方法二：使用Z-score标准化
from sklearn import preprocessing
breast_cancer_zscore = pd.DataFrame(preprocessing.scale(breast_cancer),\
                                    columns = breast_cancer.columns)
breast_cancer_zscore.head()
#划分训练集和测试集
breast_cancer_zscore_train,breast_cancer_zscore_test,\
breast_cancer_train_labels,breast_cancer_test_labels \
= cross_validation.train_test_split(breast_cancer_zscore,y,test_size=0.3,random_state=0)
#模型训练
knn_model_z = KNeighborsClassifier(n_neighbors =5) #k近邻数为5
knn_model_z.fit(breast_cancer_zscore_train,breast_cancer_train_labels)

#模型预测
breast_cancer_test_pred_z = knn_model_z.predict(breast_cancer_zscore_test)

#性能评估
accuracy_z = metrics.accuracy_score(breast_cancer_test_labels,breast_cancer_test_pred_z)
confusion_mat_z = metrics.confusion_matrix(breast_cancer_test_labels,breast_cancer_test_pred_z)

print("k = 5")
print("\t正确率：",'%.2f'%(accuracy_z*100) + "%")
print("\t假阴性：",confusion_mat_z[0,1])
print("","\t假阳性：",confusion_mat_z[1,0])

'''
k = 5
        正确率： 95.91%
        假阴性： 1
        假阳性： 6
        
在用zscore标准化的模型中，假阳性数量反而增加了在k=5时，
这说明zscore标准化更差比minmax标准化
'''





















