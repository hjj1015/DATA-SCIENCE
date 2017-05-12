# -*- coding: utf-8 -*-
"""


@author: Administrator
"""

import pandas as pd

sms_raw = pd.read_csv("sms_spam.csv")
sms_raw.describe()
sms_raw.dtypes

#当前特征type是个字符串向量，需将其转换成一个因子变量，1表示垃圾短信，0表示非垃圾短信
sms_raw['type'] = pd.Series(sms_raw['type'].factorize()).iloc[0]

#观察变量的分布情况
sms_raw.groupby('type').count()

import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn import metrics

def tolower(text):
    return text.lower()

def removePunctuation(text):
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.translate(str.maketrans('','','1234567890'))
    return text

sms_raw['text'] = sms_raw['text'].map(removePunctuation).map(tolower)
count_vect = CountVectorizer(stop_words="english",decode_error="ignore")
sms_counts = count_vect.fit_transform(sms_raw['text'])
sms_counts.shape

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words="english",decode_error='ignore',min_df=5)
sms_tfidf = tfidf_vect.fit_transform(sms_raw['text'])
sms_tfidf.shape

sms_trainx = sms_tfidf[0:4175,]
sms_trainy = sms_raw['type'][0:4175]
sms_testx = sms_tfidf[4176:5567,]
sms_testy = sms_raw['type'][4176:5567]

#可视化文本数据---词云
WordCloud(sms_raw['text'])
WordCloud(sms_raw['text'][sms_raw['type']==1])
WordCloud(sms_raw['text'][sms_raw['type']==0])

#训练模型
from sklearn.naive_bayes import MultinomialNB
sms_classifier = MultinomialNB().fit(sms_trainx,sms_trainy)

#评估模型性能
sms_test_pred = sms_classifier.predict(sms_testx)
metrics.confusion_matrix(sms_testy,sms_test_pred)
print(metrics.classification_report(sms_testy,sms_test_pred))


