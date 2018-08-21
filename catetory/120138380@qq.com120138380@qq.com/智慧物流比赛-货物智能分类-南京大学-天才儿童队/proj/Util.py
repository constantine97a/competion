# -*- coding: utf-8 -*-

# @Time    : 2018/8/13
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Util.py

import xlrd

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
punc = ['`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
        '[', ']', '{', '}', '\\', '|', ';', ':', '\'', '"', ',', '<', '.', '>', '/', '?',
        '·', '！', '￥', '……', '…', '（', '）', '——', '—',
        '【', '】', '、', '；', '：', '‘', '’', '“', '”', '，', '《', '。', '》', '？']

vocab_num = 510
sen_len = 4

def split(line):
    s = []
    for w in line:
        if w != '' and w != '克' and w not in alphabet and w not in punc:
            s.append(w)
    return s

def get_vocab_dist():
    fr = open('dataset/words_num.csv', encoding='utf-8')
    vocab_dict = {}
    for i in range(vocab_num):
        vocab_dict[fr.readline().strip().split('\t')[0]] = i
    fr.close()
    return vocab_dict

def get_label_dict():
    workbook = xlrd.open_workbook('dataset/运输货物的分类和代码-细分.xlsx')
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(3, table.nrows)]
    first_level_dict = {}
    second_level_dict = {}
    third_level_dict = {}
    label_dict1 = {}
    label_dict2 = {}
    label_dict3 = {}
    label1 = 0
    label2 = 0
    label3 = 0
    for line in lines:
        v1 = line[0].value.strip()
        if v1 != '':
            if '钢 铁' == v1:
                v1 = '钢铁'
            first_level_dict[v1] = label1
            label_dict1[label1] = v1
            label1 += 1
        v2 = line[1].value.strip()
        if v2 != '':
            second_level_dict[v2] = label2
            label_dict2[label2] = v2
            label2 += 1
        v3 = line[2].value.strip()
        if v3 != '' and v3 != '\\':
            third_level_dict[v3] = label3
            label_dict3[label3] = v3
            label3 += 1
    second_level_dict['石油焦'] = label2
    label_dict2[label2] = '石油焦'
    label2 += 1
    second_level_dict['其它'] = label2
    label_dict2[label2] = '其它'
    label2 += 1
    third_level_dict['其它'] = label3
    label_dict3[label3] = '其它'
    label3 += 1
    return first_level_dict, second_level_dict, third_level_dict, label_dict1, label_dict2, label_dict3

def get_word_index(line, vocab_dict):
    index = []
    for w in line:
        index.append(vocab_dict[w] if w in vocab_dict.keys() else vocab_num + 1)
    length = len(index)
    if length < sen_len:
        index.extend([vocab_num] * (sen_len - length))
    elif length > sen_len:
        index = index[ : sen_len]
    return index

def get_word_num():
    workbook = xlrd.open_workbook('dataset/货物分类训练数据.xlsx')
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(1, table.nrows)]
    words = {}
    for line in lines:
        line = split(line[0].value.strip())
        for w in line:
            if w not in words.keys():
                words[w] = 0
            words[w] += 1
    words = sorted(words.items(), key=lambda d: d[1], reverse=True)
    fw = open("dataset/words_num.csv", mode='a', encoding='UTF-8')
    for w in words:
        fw.write(w[0] + '\t' + str(w[1]) + '\n')
    fw.close()
