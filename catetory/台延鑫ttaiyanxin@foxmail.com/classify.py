# coding: utf-8

# In[65]:


import jieba.posseg
import numpy as np
import pandas as pd
from sklearn import neighbors

train = pd.read_excel(u'货物分类训练数据.xlsx')
pre = pd.read_excel(u'需要分类货物数据.xlsx')

train = train.drop_duplicates()
train = train.reset_index(drop=True)


# train = train[1:1000]
# print(train.shape)


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


train['sort_name'] = train[u'货物全称'].apply(lambda x: format_str(x))
pre['sort_name'] = pre[u'货物名称'].apply(lambda x: format_str(x))

train = train[train[u'一级分类'] != u'待定']
train = train.reset_index(drop=True)


def clean(x):
    x = x.replace(u"其他", u"其它")
    return x


train[u'一级分类'] = train[u'一级分类'].apply(lambda x: clean(x))


def my_test(a, b, c):
    if b != b or b == '\\':
        return False
    if a != a or a == '\\':
        return False
    if c != c or c == '\\':
        return False
    return True


train['is_right'] = train.apply(lambda x: my_test(x[u'一级分类'], x[u'二级分类'], x[u'三级分类']), axis=1)
train = train[train['is_right'] == True]
train = train.reset_index(drop=True)

# 思路：多标签、多种类
# 监督学习：分词+相似度匹配+K近邻？分词+文本特征+训练（这个因为标签太多很难多分类，有的只有1种）？

# 先直接分词，然后实验算法（随机划分，先打好基础，观察准确率） 一级二级与三级


import jieba

# le = preprocessing.LabelEncoder()
# le.fit(train['一级分类'])
# train['1_kind'] = le.transform(train['一级分类'])
# le.fit(train['二级分类'].astype(str))
# train['2_kind'] = le.transform(train['二级分类'].astype(str))
# le.fit(train['三级分类'].astype(str))
# train['3_kind'] = le.transform(train['三级分类'].astype(str))

# count1 = pd.DataFrame(train.groupby(['1_kind'])['一级分类'].count().reset_index())
# count1.columns = ['1_kind', 'count1']
# train = train.merge(count1, on='1_kind', how='left')
# count1 = pd.DataFrame(train.groupby(['2_kind'])['一级分类'].count().reset_index())
# count1.columns = ['2_kind', 'count2']
# train = train.merge(count1, on='2_kind', how='left')
# count1 = pd.DataFrame(train.groupby(['3_kind'])['一级分类'].count().reset_index())
# count1.columns = ['3_kind', 'count3']
# train = train.merge(count1, on='3_kind', how='left')

# train=train.reset_index(drop=True)

jieba.add_word(u'热镀')
jieba.add_word(u'方管')
jieba.add_word(u'盘螺')
jieba.add_word(u'乌冬面')
jieba.add_word(u'栈板')
jieba.add_word(u'喷塑')
jieba.add_word(u'潘婷')
jieba.add_word(u'碧浪')
jieba.add_word(u'汰渍')
jieba.add_word(u'麦格尼飞')
jieba.add_word(u'卷板')
jieba.add_word(u'郎牌')
jieba.add_word(u'原纸')
jieba.add_word(u'伸性纸')
jieba.add_word(u'版纸')
jieba.add_word(u'薄本纸')
jieba.add_word(u'淮盐牌')
jieba.add_word(u'化学浆')
jieba.add_word(u'宝光')
jieba.add_word(u'恒大')
jieba.add_word(u'分路器')
jieba.add_word(u'管坯钢')


def all(x):
    alist = list(jieba.cut(x, cut_all=True))
    blist = list(x)
    last = []
    #     a=list(jieba.posseg.cut(x))
    # # #     print(a)
    # #     for i in a:
    # #         print(i.word,i.flag)
    # # #     print(a)
    # #     print(i,a)
    #     try:
    #         last=[[i.word for i in a if 'n' in i.flag][-1]]
    #     except:
    #         print(a)
    #         last=[a[-1].word]

    return alist + blist + last


train = train.reset_index(drop=True)
# train['word_cut']=train['sort_name'].apply(lambda x:list(jieba.cut(x,cut_all=True)))
# train['word_cut']=train['word_cut'].apply(lambda x:x+list(train['sort_name']))
train['word_cut'] = train['sort_name'].apply(lambda x: all(x))
pre['word_cut'] = pre['sort_name'].apply(lambda x: all(x))
# for i in train_many['sort_name']:
#     a=list(jieba.posseg.cut(i))
# #     print(a)
# #     print(i,list(jieba.posseg.cut(i)))
#     print([i.word for i in a if 'n' in i.flag][-1])
# vectorizer = CountVectorizer()
# #计算个词语出现的次数
# X = vectorizer.fit_transform(corpus)
allword = []

for i in train.word_cut:
    for j in i:
        allword.append(j)
for i in pre.word_cut:
    for j in i:
        allword.append(j)
allword = list(set(allword))


def encode(x):
    encodelist = [True if i in x else False for i in allword]
    #     lastlist=[True if i in x[-1] else False for i in allword]
    lastlist = []
    return encodelist + lastlist


train['word_dict'] = train['word_cut'].apply(lambda x: encode(x))
train['word_dict'] = train['word_dict'].apply(lambda x: list(x))
pre['word_dict'] = pre['word_cut'].apply(lambda x: encode(x))
pre['word_dict'] = pre['word_dict'].apply(lambda x: list(x))
alist = []
for i in train['word_dict']:
    alist.append(i)
# for i in pre['word_dict']:
#     alist.append(i)
wordarray = np.array(alist)
# import numpy as np
# a=np.array(train_many['word_cut'])
tree = neighbors.BallTree(wordarray, metric='sokalsneath')
al = []
bl = []
cl = []
dl = []
el = []
fl = []
count = 0
# train_many = pd.DataFrame(train[train['count3'] != 1])
# # train_many=train_many.reset_index(drop=True)
# # train_many.to_csv('aaaa.csv')
# for m in train_many['word_dict']:
#     dist, ind = tree.query(np.array(m).reshape(1, -1), k=4)
# #     print('dis:',dist)
#     print(ind)
#     a = train['三级分类'][ind[0][1]]
# #     b = train['三级分类'][ind[0][2]]
# #     e=train['三级分类'][ind[0][3]]
#     c = train['word_cut'][ind[0][1]]
# #     d = train['word_cut'][ind[0][2]]
# #     f=train['word_cut'][ind[0][3]]
#     al.append(a)
# #     bl.append(b)
# #     cl.append(c)
# #     el.append(e)
# #     fl.append(f)
#     cl.append(c)
# #     dl.append(d)

#     count=count+1
#     if count%100==0:
#         print(count)


# In[66]:


# pre = pd.read_excel('需要分类货物数据.xlsx')
# train_many=train_many.reset_index(drop=True)
# train_many.to_csv('aaaa.csv')
for m in pre['word_dict']:
    dist, ind = tree.query(np.array(m).reshape(1, -1), k=1)
    print('dis:', dist)
    #     print(ind)
    a3 = train[u'三级分类'][ind[0][0]]
    a2 = train[u'二级分类'][ind[0][0]]
    a1 = train[u'一级分类'][ind[0][0]]
    #     b = train['三级分类'][ind[0][2]]
    #     e=train['三级分类'][ind[0][3]]
    #     c = train['word_cut'][ind[0][0]]
    #     d = train['word_cut'][ind[0][2]]
    #     f=train['word_cut'][ind[0][3]]
    al.append(a1)
    bl.append(a2)
    cl.append(a3)
    #     bl.append(b)
    #     cl.append(c)
    #     el.append(e)
    #     fl.append(f)
    #     cl.append(c)
    #     dl.append(d)

    count = count + 1
    if count % 100 == 0:
        print(count)

# In[67]:


wordarray.shape

# In[68]:


train_many.shape

# In[69]:


pre[u'一级分类'] = al
pre[u'二级分类'] = bl
pre[u'三级分类'] = cl

# train_many['pre32']=bl
# train_many['w31']=cl
# train_many['w32']=dl
# train_many['pre33']=el
# train_many['w33']=fl
# train_many['pre33']=cl


# In[70]:


# print((train_many[train_many['三级分类'] == train_many['pre31']].shape[0]) / train_many.shape[0]) 
# print((train_many[train_many['三级分类'] == train_many['pre32']].shape[0]) / train_many.shape[0]) 
# print((train_many[train_many['三级分类'] == train_many['pre33']].shape[0]) / train_many.shape[0]) 
# print((train_many[train_many['三级分类'] == train_many['pre33']].shape[0]) / train_many.shape[0]) # 正确率

# print(train_many[train_many['三级分类'] != train_many['pre3']])


# In[71]:


# train_many['right1']=(train_many['三级分类'] == train_many['pre31'])
# train_many['right2']=(train_many['三级分类'] == train_many['pre32'])
# train_many['right3']=(train_many['三级分类'] == train_many['pre33'])
# train_many['rightany']=train_many['right1']|train_many['right2']|train_many['right3']

# def my_test(a, b,c):
#     if b==c:
#         return b
#     else:
#         return a


# train_many['pre123']=train_many.apply(lambda x: my_test(x['pre31'], x['pre32'], x['pre33']), axis=1)

# df['Value'] = df.apply(lambda row: my_test(row['a'], row['c']), axis=1)


# In[72]:


# print((train_many[train_many['三级分类'] == train_many['pre123']].shape[0]) / train_many.shape[0]) #确实采用后两个的有提升


# 0.9256794231835829  S  4 投票
# 0.9286110450835908  S  1

# 0.7402022756005057 字  jaccard
# 
# 0.7331700379266751  cut all=False jacc
# 
# 0.747629582806574  cut all=True  jacc
# 
# 0.7575853350189633  字+True  jacc
# 
# 0.7492888748419722  字+False jacc
# 
# 0.7540297092288243 字+True+后面两个字  matching 
# 
# 0.7583754740834386  字+True+后面一个字  jacc
# 
# 0.8424462705436156  字+True+后面一个字  jacc  knn=4 max?
# 
# 0.9245701608430393
# 字+True+True+后面一个字  jacc  knn=2
# 
# 0.9249663259646621
# # sokalsneath  0.9307503367403533 字+True  knn=4 不融合 #
#  0.9303541716187307 字+True  jacc knn=4 不融合 
# 目前改进方法：
# 利用词性+名词尝试
# 调参：距离 与 K tfidf?
# 集成
# 添加人工词典
# 
# 
# 想法:重点突出名词 实现方式：tf加其他相似度指标
# 或者先用其他监督学习训练大分类（比较小分类（一级的）准确度，最后KNN小分类）

# In[87]:


del pre[u'word_dict']
del pre[u'sort_name']
del pre[u'word_cut']
pre.to_csv(u'pre_result.csv', index=False)
# pre.to_csv('pre_result.csv',encoding='gbk')


# In[74]:


# diff=train_many[train_many['三级分类'] != train_many['pre31']]


# In[75]:


# diff.shape


# In[76]:


# diff.to_csv('aa.csv',encoding='gbk')


# In[77]:


# diff[['right1','right2','right3','rightany']]#最后再尝试吧


# In[78]:


# lena=[len(i) for i in al]


# In[79]:


# max(lena)


# In[80]:


# diff


# In[81]:


# import jieba.posseg
# a=list(jieba.posseg.cut('漂亮的太阳牌的小玩具'));a


# In[82]:


# for i in train_many['sort_name']:
#     a=list(jieba.posseg.cut(i))
# #     print(a)
# #     print(i,list(jieba.posseg.cut(i)))
#     print([i.word for i in a if 'n' in i.flag][-1])


# In[83]:


# a[0].word


# In[84]:


# a[0].flag
