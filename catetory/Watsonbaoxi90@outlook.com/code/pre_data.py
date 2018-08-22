#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import random
import re
from collections import defaultdict

import wubi
import xlrd

pwd = os.getcwd()

raw_standard_path = pwd + '/data/运输货物的分类和代码-细分.xlsx'
raw_pre_path = pwd + '/data/data.xlsx'

pre_1 = pwd + '/data/pre1.conll'
pre_2 = pwd + '/data/class.conll'

# 先处理货物分类训练数据1to3


rule = re.compile("[^\u4e00-\u9fa5]")

before_texts = 'aaa'


def transfer_wubi(text):
    code = wubi.get(text, 'cw')
    return list(code)[0]


def iter_item(raw_item):
    texts = rule.sub('', str(raw_item))
    if texts != '':
        for t in list(texts):
            yield t


def flase_gen(raw_item):
    texts = rule.sub('', str(raw_item))
    try:
        if texts != '':
            return random.choice(list(t_dict[texts]))
        else:
            return random.choice(list(class_set))
    except:
        return random.choice(list(class_set))


def pre_process(raw_pre_path, class_set):
    raw_file = xlrd.open_workbook(raw_pre_path)
    raw_table = raw_file.sheets()[0]
    nrows = raw_table.nrows
    class_list = list(class_set)

    for i in (x + 1 for x in range(nrows - 1)):
        # 先处理cell[0]
        item_list = []
        raw_item = raw_table.cell(i, 0)
        for text in iter_item(raw_item):
            item_list.append(text)
            code = transfer_wubi(text)
            item_list.append(code)
        item = ','.join(item_list)
        pre_file_1.write(item + '\n')
    for text_c in class_list:
        c_list = []
        for t in iter_item(text_c):
            c_list.append(t)
            code = transfer_wubi(t)
            c_list.append(code)
        class_str = ','.join(c_list)
        pre_file_2.write(class_str + '\n')


def standard_process(raw_standard_path):
    first_class = set()
    class_set = set()
    second_class = defaultdict(set)
    third_class = defaultdict(set)
    raw_stand_file = xlrd.open_workbook(raw_standard_path)
    stand_table = raw_stand_file.sheets()[0]
    nrows = stand_table.nrows
    ncols = stand_table.ncols
    table = [[0 for i in range(nrows)] for i in range(3)]
    for x in (x + 3 for x in range(nrows - 3)):

        add_list = [[], [], []]
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
    for i in range(nrows - 3):
        dict_text_1 = rule.sub('', table[0][i])
        first_class.add(dict_text_1)
        dict_text_2 = rule.sub('', table[1][i])
        second_class[dict_text_1].add(dict_text_2)
        dict_text_3 = rule.sub('', table[2][i])
        third_class[dict_text_2].add(dict_text_3)
    return class_set


if __name__ == '__main__':
    pre_file_1 = open(pre_1, 'w')
    pre_file_2 = open(pre_2, 'w')
    # first_class, second_class, third_class = standard_process(raw_standard_path)
    class_set = standard_process(raw_standard_path)
    pre_process(raw_pre_path, class_set)
    pre_file_1.close()
