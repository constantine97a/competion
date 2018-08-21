#!/usr/bin/env python
# -*- coding: utf-8 -*-

#_author_='songyan'
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import  jieba
import pandas as pd
import numpy as np

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

#train_content,test_content,train_lable,test_lable=train_test_split(content,lable,test_size=0.4,random_state=0)

def predicted(train_content,train_lable,test_content):
  vetorizer=CountVectorizer()
  tfidtransformer=TfidfTransformer()
  tfid1=tfidtransformer.fit_transform(vetorizer.fit_transform(train_content))

  cls=MultinomialNB(alpha=0.01)
  cls=cls.fit(tfid1,train_lable)

  print('训练集',cls.score(tfid1,train_lable))

  new_tfid=tfidtransformer.transform(vetorizer.transform(test_content))
  predicted=cls.predict(new_tfid)
  return predicted

def adjust_param(tfid1,new_tfid,train_lable,test_lable):
    alphas=np.logspace(-2,5,num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = MultinomialNB(alpha=alpha)
        cls.fit(tfid1, train_lable)
        train_scores.append(cls.score(tfid1, train_lable))
        test_scores.append(cls.score(new_tfid, test_lable))
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(alphas,train_scores,label=u"Training Score")
    ax.plot(alphas,test_scores,label=u"Testing Score")
    plt.legend()
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score")
    ax.set_ylim(0,1.0)
    ax.set_title("MultinomialNB")
    ax.set_xscale("log")
    plt.show()


if __name__=="__main__":
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
    frame.to_csv('MultinomialNB.csv')





