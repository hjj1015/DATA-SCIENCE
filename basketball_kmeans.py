# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:21:34 2017
任务: kmeans分类
数据集介绍：样本数96个，5个特征
assists_per_minute：平均每分钟的助攻次数
height： 运动员身高
time_played：运动员打球时间
age ： 远动员年龄
points_per_minute: 平均每分钟的得分数

@author: Administrator
"""

import numpy as np
import pandas as pd
basketball = pd.read_csv("basketball_kmeans.txt")
basketball.head()
'''检测缺失值数量'''
basketball_value_ravel = basketball.values.ravel()
print("缺失值数目：",len(basketball_value_ravel[basketball_value_ravel==np.nan]))
#  缺失值数目： 0
basketball.describe()
basketball.height.value_counts()
basketball.age.value_counts()
'''发现需要进行标准化，这里z标准化'''
from sklearn import preprocessing
columns_list = ['assists_per_minute', 'height', 'time_played', 'age',
                'points_per_minute']
basketball_zscore = pd.DataFrame(preprocessing.scale(basketball[columns_list]),
                                columns=basketball[columns_list].columns)
'''观察是否已经标准化'''
basketball_zscore.describe()  

'''模型训练'''
from sklearn.cluster import KMeans
basketball_cluster_model = KMeans(n_clusters=5)
basketball_cluster_model.fit(basketball_zscore)
'''
----------------------------------------------------------------
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=5, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0) 
n_clusters:类别数目  init：指定初始化类中心的策略
n_init：指定kmeans算法运行的次数。每次都选择一组不同的初始化均值向量，
        最终算法选择最佳的分类簇作为结果
max_iter：单轮kmeans算法运行的最大迭代次数，算法总的迭代次数为n_init* max_iter
precompute_distances:
    指点是否提前计算好样本之间的距离(提前计算距离需要更多内存但算的块)
    auto 如果n_samples*n_clusters>12 million ,则不提前计算
    True 总是提前计算
    False 总是不提前计算
tol: 浮点数，指点算法收敛的阈值
n_jobs: 指定任务并行时指定的CPU数量，如果为-1则使用所有可能的CPU
verbose: 如果为0，则不输出日志信息；如果为1，则每隔一段时间输出一次日志信息；
        如果大于1，则打印日志信息频繁
random_state: 随机数生成器，为None时使用默认的随机数生成器
copy_x : 主要用于precompute_distances=True的情况
        在precompute_distances=True时，copy_x=True，则预计算距离时并不修改数据
        copy_x=False,则预计算距离时，会修改原始数据用于节省内存，然后当算法结束的时候将原数据返还
--------------------------------------------------------------
下面是返回的属性值：
cluster_centers_ ;给出分类簇的均值向量
labels_: 给出了每个样本所属的簇的标记
inertia_ : 给出了每个样本距离它们各自最近的簇中心的距离之和
---------------------------------------------------------------
'''
'''观察聚类结果每个类的数量'''
basketball_clusters = pd.Series(basketball_cluster_model.labels_)
basketball_clusters.value_counts().sort_index()
'''
Out[91]: 
0    24
1    15
2    15
3    19
4    23
'''
#查看类中心
centers = pd.DataFrame(basketball_cluster_model.cluster_centers_,
                       columns = basketball_zscore.columns)
'''返回的是一个类质心的数据框,下面分析每个聚类中心的含义'''
centers_t = centers.T
centers_t.columns = ["cluster_0","cluster_1","cluster_2",
                     "cluster_3","cluster_4"]
centers_t["cluster_0"].sort(ascending = False,inplace=False).head() 
'''
降序显示每个类中心特征
Out[95]: 
assists_per_minute    0.470442
age                  -0.576261
points_per_minute    -0.596567
height               -0.613801
time_played          -0.861576
Name: cluster_0, dtype: float64

这说明该类运动员是：年轻化，身高中下，打球时间不长但是场均助攻却属于中上的运动员
'''                   
centers_t["cluster_1"].sort(ascending = False,inplace=False).head() 
'''
points_per_minute     1.430713
time_played           1.075582
height                0.778685
age                   0.562406
assists_per_minute   -0.630711
Name: cluster_1, dtype: float64
该类运动员属于：资格老，得分高，打球时间长，助攻却不多的一类，后面的省略
'''
print("Sum_center_distance =%.2f" %basketball_cluster_model.inertia_)

