#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os
import random
import re
from collections import defaultdict

import wubi
import xlrd

pwd = os.getcwd()

raw_train_path = pwd + '/data/货物分类训练数据.xlsx'
raw_standard_path = pwd + '/data/运输货物的分类和代码-细分.xlsx'
raw_need_class_path = pwd + '/data/需要分类货物数据.xlsx'
train_word2vce_file = pwd + '/data/train_word2vec.conll'
train1 = pwd + '/data/train'
class1 = pwd + '/data/class'
class3 = pwd + '/data/class3'
dev1 = pwd + '/data/dev'
vocab_path = pwd + '/data/vocab.txt'
vec_path = pwd + '/data/vec.txt'
t_dict = defaultdict(set)
class_set = set()
# 先处理货物分类训练数据1to3


rule = re.compile("[^\u4e00-\u9fa5]")

before_texts = 'aaa'

unk = 'uuunnnkkk'
pad_id = '<pad>'

voc = []
vec = []


def vocab_parse(path, vocab):
    for line in open(path):
        vocab.append(line.strip())


def vec_parse(path, vec):
    for line in open(path):
        vector = line.strip()
        vec.append(list(map(float, vector.split())))


vocab_parse(vocab_path, voc)
vec_parse(vec_path, vec)


def find_index(vocab, word):
    if word in vocab:
        return vocab.index(word)
    else:
        code = transfer_wubi(word)
        if code in vocab:
            return vocab.index(word)
        else:
            return vocab.index(unk)


def sentence_token(sentence):
    newsentence = []
    vocab = voc
    for word in sentence:
        newsentence.append(find_index(vocab, word))
    return newsentence


def transfer_wubi(text):
    code = wubi.get(text, 'cw')
    c = list(code)
    return c[0]


def iter_item(raw_item):
    texts = rule.sub('', str(raw_item))
    global before_texts
    if texts != '':
        before_texts = texts
        return texts
    else:
        return before_texts


def flase_gen(raw_item):
    texts = rule.sub('', str(raw_item))
    try:
        if texts != '':
            return random.choice(list(t_dict[texts]))
        else:
            return random.choice(list(class_set))
    except:
        return random.choice(list(class_set))


def generate_traindata(train_dict, fake_list):
    encoder_inputs = []
    decoder_inputs = []
    simi = []
    for item in train_dict:
        item_list = []
        for t in list(item):
            item_list.append(t)
            code = transfer_wubi(t)
            item_list.append(code)
        item_sentence = sentence_token(item_list)
        for true_class in train_dict[item]:
            sentence = []
            for t in list(true_class):
                sentence.append(t)
                code = transfer_wubi(t)
                sentence.append(code)
            encoder_inputs.append(item_sentence)
            decoder_inputs.append(sentence_token(sentence))
            simi.append(1)
        for fake_class in random.sample(fake_list, min(8,len(fake_list))):
            if fake_class not in train_dict[item]:
                sentence = []
                for t in list(fake_class):
                    sentence.append(t)
                    code = transfer_wubi(t)
                    sentence.append(code)
                encoder_inputs.append(item_sentence)
                decoder_inputs.append(sentence_token(sentence))
                simi.append(0)
    return encoder_inputs, decoder_inputs, simi


def train_process(raw_train_path, all_dict):
    raw_file = xlrd.open_workbook(raw_train_path)
    raw_table = raw_file.sheets()[0]
    nrows = raw_table.nrows
    dev_dict = dict()

    for i in (x + 1 for x in range(nrows - 1)):
        # 先处理cell[0]

        c_list = []
        raw_item = raw_table.cell(i, 0)
        texts = rule.sub('', str(raw_item))
        if texts != '':
            # 再处理cell[1:3]
            for n in [1, 2, 3]:
                raw_class_n = raw_table.cell(i, n)
                text = iter_item(raw_class_n)
                c_list.append(text)
            if i % 10 == 0:
                dev_dict[texts] = c_list
            else:
                all_dict[texts] = c_list
    return all_dict, dev_dict


def standard_process(raw_standard_path):
    raw_stand_file = xlrd.open_workbook(raw_standard_path)
    stand_table = raw_stand_file.sheets()[0]
    nrows = stand_table.nrows
    ncols = stand_table.ncols
    table = [[0 for i in range(nrows)] for i in range(3)]
    all_dict = dict()
    class_set = set()
    class_set_3 = set()

    for x in (x + 3 for x in range(nrows - 3)):
        true_list = []
        texts = stand_table.row_values(x)
        item_text = texts[3]
        for i in [0, 1, 2]:
            if texts[i] == '':
                texts[i] = table[i][x - 4]
            elif texts[i] == '\\':
                texts[i] = texts[i - 1]
            table[i][x - 3] = texts[i]
            dict_text = rule.sub('', texts[i])
            class_set.add(dict_text)
            true_list.append(dict_text)
            if i == 2:
                class_set_3.add(dict_text)
        if item_text != '':
            item_text = item_text.replace(u'、', ' ').replace(u'）', ' ').replace(u'（', ' ').replace(u';', ' ')
            item_text = item_text.strip().split(' ')
            for t in item_text:
                t = rule.sub('', t)
                if t != '':
                    all_dict[t] = true_list
    return all_dict, class_set, class_set_3


if __name__ == '__main__':

    all_dict, class_set, class_set_3 = standard_process(raw_standard_path)
    all_dict, dev_dict = train_process(raw_train_path, all_dict)
    with open(train1, 'w') as f, open(class1, 'w') as c, open(class3, 'w') as c3, open(dev1, 'w') as d1:
        train_str = json.dumps(all_dict, ensure_ascii=False)
        dev_str = json.dumps(dev_dict, ensure_ascii=False)
        f.write(train_str)
        d1.write(dev_str)
        for i in class_set:
            c.write(i + '\n')
        for i in class_set_3:
            c3.write(i + '\n')
    '''
    ff = open(train1,'r')
    for line in ff:
        str_f = json.loads(line)
    ff.close()
    '''
