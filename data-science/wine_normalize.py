# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 15:32:30 2017

@author: Administrator
"""

import pandas as pd 
import numpy as np
from sklearn import preprocessing
import seaborn as sns


wine = pd.read_csv("E:\datasets\wine.txt",sep=',',header=None)
'''这里结果展示，只取两个特征变量列，一个label列'''
wine = wine.iloc[:,0:3]
#修改列名
wine.columns = ["Class","Alcohol","Malic acid"]
#查看前五行数据
wine.head()

'''观察数据发现，两个特征的取值范围不同，需要进行标准化'''
#0-1标准化,只需要对两个特征列进行标准化即可
minmax_scale = preprocessing.MinMaxScaler().fit(wine[["Alcohol","Malic acid"]])
df_minmax = minmax_scale.transform(wine[["Alcohol","Malic acid"]])

#z-score标准化
std_scale = preprocessing.StandardScaler().fit(wine[["Alcohol","Malic acid"]])
df_std = std_scale.transform(wine[["Alcohol","Malic acid"]])

'''查看两个特征变量不同标准化的取值特点'''
#0-1标准化，两个特征变量的最大值和最小值
print('Min-value after 0-1 scaling:\nAlcohol={:.2f},Malic acid={:.2f}'
      .format(df_minmax[:,0].min(),df_minmax[:,1].min()))


df_minmax = pd.DataFrame(df_minmax)
df_minmax.columns = ["Alcohol","Malic acid"]
df_minmax["Class"] = "0-1 Scaling"

df_std = pd.DataFrame(df_std)
df_std.columns = ["Alcohol","Malic acid"]
df_std["Class"] = "z-score"

wine_contrast = wine[["Alcohol","Malic acid"]]
wine_contrast["Class"] = "input"

wine_contrast = pd.concat([wine_contrast,df_minmax,df_std])

wine_contrast.to_csv("wine_contrast.csv",index=False)

sns.pairplot(hue="Class",data=wine_contrast,x_vars="Alcohol",
             y_vars="Malic acid",size=5,aspect=2)
'''默认高度size = 2.5，宽度aspect = 1'''
sns.plt.show()
#将图片保存
sns.plt.savefig("wine.png")

'''按类别1,2,3画出原数据分布'''
sns.pairplot(hue="Class",data=wine,x_vars="Alcohol",
             y_vars="Malic acid",size=5,aspect=2)

df_minmax["Class"] = wine["Class"]
'''按类别1,2,3画出0-1标准化数据分布'''
sns.pairplot(hue="Class",data=df_minmax,x_vars="Alcohol",
             y_vars="Malic acid",size=5,aspect=2)

df_std["Class"] = wine["Class"]
'''按类别1,2,3画出z-score标准化数据分布'''
sns.pairplot(hue="Class",data=df_std,x_vars="Alcohol",
             y_vars="Malic acid",size=5,aspect=2)










