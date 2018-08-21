#!/usr/bin/env python
# -*- coding: utf-8 -*-

#_author_='songyan'
import csv
import jieba
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd


def readtrain(path):
    content_train=pd.read_csv(filepath_or_buffer=path,sep=',')["货物全称"].values
    level1_train=pd.read_csv(filepath_or_buffer=path,sep=',')['一级分类'].values
    level2_train = pd.read_csv(filepath_or_buffer=path, sep=',')['二级分类'].values
    level3_train = pd.read_csv(filepath_or_buffer=path, sep=',')['三级分类'].values
    train=[content_train,level1_train,level2_train,level3_train]
    return train

def segmentword(cont):
    c=[]
    for i in cont:
        a=list(jieba.cut(i))
        b=''.join(a)
        c.append(b)
    return c

#train_content,test_content,train_lable,test_lable=train_test_split(content,lable,test_size=0.25,random_state=0)

def predicted(train_content,train_lable,test_content):
    vectorizer = CountVectorizer()
    tfidtransformer = TfidfTransformer()
    train_tfidf = tfidtransformer.fit_transform(vectorizer.fit_transform(train_content))
    print(train_tfidf.shape)
    text_clf = Pipeline(
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC(C=100, gamma=0.001, kernel='linear'))])
    text_clf = text_clf.fit(train_content, train_lable)
    print(text_clf.score(train_content, train_lable))
    predicted = text_clf.predict(test_content)
    return predicted



if __name__=='__main__':
    frame=pd.DataFrame()
    path='train_data.csv'
    train = readtrain(path)
    #三级分类
    train_content = segmentword(train[0].astype(str))
    train_lable = train[3].astype(str)
    test_content = pd.read_csv(filepath_or_buffer='classify_data.csv', sep=',')["货物名称"].values
    frame['货物名称']=test_content
    frame['三级分类']=predicted(train_content,train_lable,test_content)
    #二级分类
    train_content = segmentword(train[3].astype(str))
    train_lable = train[2].astype(str)
    test_content=frame['三级分类']
    frame['二级分类']=predicted(train_content,train_lable,test_content)
    #一级分类
    train_content = segmentword(train[2].astype(str))
    train_lable = train[1].astype(str)
    test_content = frame['二级分类']
    frame['一级分类'] = predicted(train_content, train_lable, test_content)
    #保存结果
    frame.to_csv('SVM.csv')





