# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:57:07 2017

@author: Administrator
"""

import pandas as pd 
import numpy as np

banana = pd.read_csv("banana.csv")
#删除第三列数据
banana.drop("Unnamed: 2",axis=1,inplace=True)
banana.head()

##############################
'''等距离散化'''
banana.max()
'''
At1    2.81
At2    3.19
dtype: float64
'''
banana.min()
'''
Out[7]: 
At1   -3.09
At2   -2.39
dtype: float64
'''
#等距生成长度为25的数组
bins1 = np.linspace(-3.09,2.81,25)
bins2 = np.linspace(-2.39,3.19,25)

#取出第一列数据
At1 = banana.iloc[:,0]

#取出第二列数据
At2 = banana.iloc[:,1]

At1_d = np.digitize(At1,bins1)   #digitize离散化函数
At2_d = np.digitize(At2,bins2)   #digitize离散化函数

banana_d = pd.concat([pd.DataFrame(At1_d),pd.DataFrame(At2_d)]
                     ,axis=1)
banana_d.columns = ["At1","At2"]
banana_d.head()

####################################
'''等频离散化,等频离散化函数调用pandas中qcut函数'''
'''生成2种区间用于离散化数据'''
bins1 = pd.qcut(At1,2)  
bins2 = pd.qcut(At2,2)

print(pd.value_counts(bins1))
print(pd.value_counts(bins2))

bins1 = np.array([-3.09, -0.0152,2.81])
bins2 = np.array([-2.39, -0.0372,3.19])

At1_d = np.digitize(At1,bins1)   #digitize离散化函数
At2_d = np.digitize(At2,bins2)   #digitize离散化函数

banana_d = pd.concat([pd.DataFrame(At1_d),pd.DataFrame(At2_d)]
                     ,axis=1)
banana_d.columns = ["At1","At2"]
banana_d.head()