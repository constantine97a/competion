#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean raw data
"""

import os
import time
import re 
import wubi
import xlrd
import itertools
import numpy as np
from gensim.models import word2vec

pwd = os.getcwd()

raw_train_path = pwd + '/data/货物分类训练数据.xlsx'
raw_standard_path = pwd + '/data/运输货物的分类和代码-细分.xlsx'
raw_need_class_path = pwd + '/data/需要分类货物数据.xlsx'
train_word2vce_file = pwd + '/data/train_word2vec.conll'
model_path = pwd + '/model/word2vec_model'
vocab_path = pwd + '/data/vocab.txt'
vec_path = pwd + '/data/vec.txt'
embedding_size = 50
#先处理货物分类训练数据1to3



rule = re.compile("[^\u4e00-\u9fa5]")

before_texts = 'aaa'
def transfer_wubi(text):
    code = wubi.get( text , 'cw' )
    return code

def iter_item(raw_item):
    texts = rule.sub( '' , str(raw_item) )
    global before_texts
    if texts != '':
        before_texts = texts
        for t in list(texts):
            yield t
    else:
        for t in list(before_texts):
            yield t
def train_process(raw_train_path):
    raw_file = xlrd.open_workbook(raw_train_path)
    raw_table = raw_file.sheets()[0]
    nrows = raw_table.nrows

    for i in ( x+1 for x in range(nrows-1)):
        #先处理cell[0]
        item_list = []
        raw_item = raw_table.cell( i , 0 )
        for text in iter_item(raw_item):
            item_list.append(text)
            code = transfer_wubi( text )
            item_list.append(list(code)[0])
        #再处理cell[1:3]
        for n in [1,2,3]:
            write_list=[]
            write_list = item_list[ : ]
            raw_class_n = raw_table.cell( i , n )
            for text in iter_item(raw_class_n):
                write_list.append(text)
                code = transfer_wubi(text)
                write_list.append(list(code)[0])
            train_file.write(" ".join(write_list)+'\n')

def standard_process(raw_standard_path):
    raw_stand_file = xlrd.open_workbook(raw_standard_path)
    stand_table = raw_stand_file.sheets()[0]
    nrows = stand_table.nrows
    ncols = stand_table.ncols
    table = [[ 0 for i in range(nrows)] for i in range(3)]
    for x in ( x+3 for x in range(nrows-3)):

        add_list = [[],[],[]]
        texts = stand_table.row_values(x)
        item_text = texts[3]
        for i in [0,1,2]:
            if texts[i] == '':
                texts[i] = table[i][x-4]
            table[i][x-3] = texts[i]
            if texts[i] == '\\':
                texts[i] = texts[i-1]
                table[i][x-3] = texts[i]
            for t in list(rule.sub( '' , texts[i] ) ):
                add_list[i].append(t)
                code = transfer_wubi(t)
                add_list[i].append(list(code)[0])


        if item_text != '':
            item_text = item_text.replace(u'、',' ').replace(u'）',' ').replace(u'（', ' ').replace(u';', ' ')
            item_text = item_text.strip().split(' ')
            for t in item_text:
                t = rule.sub( '' , t )
                write_list = []
                for tt in list(t):
                    write_list.append(tt)
                    code = transfer_wubi(tt)
                    write_list.append(list(code)[0])
                for i in [0,1,2]:
                    write_list.extend(add_list[i])
                    train_file.write(" ".join(write_list)+'\n')
        else:
            write_list = list(itertools.chain.from_iterable(add_list))
            train_file.write(" ".join(write_list)+'\n')

def emb(filepath, modelpath, vocab, vec):
    sentences = word2vec.Text8Corpus(filepath)
    model = word2vec.Word2Vec(
        sentences, size=embedding_size, window=30 ,min_count=1, max_vocab_size=100000)
    model.save(modelpath)
    vocab_f = open(vocab, 'w+')
    vec_f = open(vec, 'w+')
    model = word2vec.Word2Vec.load(modelpath)
    all_words = set()
    for line in open(filepath):
        words = line.split(" ")
        for word in words:
            word = word.strip()
            if word != '':
                all_words.add(word)
    for word in all_words:
        try:
            vector = model[word]
            v = ' '.join(str(num) for num in vector)
            vec_f.writelines(v + '\n')
            vocab_f.writelines(word + '\n')
        except:
            pass
    random_vec = ' '.join(str(num) for num in (-1 + 2 *
                                               np.random.random(embedding_size)))

    vocab_f.write(unk + '\n')
    vec_f.write(random_vec + '\n')
    vocab_f.write(pad + '\n')
    vec_f.write(random_vec + '\n')

    vocab_f.close()
    vec_f.close()

if __name__ == '__main__':
    train_file = open(train_word2vce_file , 'w')
    train_process(raw_train_path)
    standard_process(raw_standard_path)
    train_file.close()
    unk = 'uuunnnkkk'
    pad = '<pad>'
    emb(train_word2vce_file, model_path, vocab_path, vec_path)   

