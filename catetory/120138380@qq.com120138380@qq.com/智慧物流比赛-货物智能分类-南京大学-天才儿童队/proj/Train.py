# -*- coding: utf-8 -*-

# @Time    : 2018/8/12
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Train.py

import xlrd
import numpy as np
from Learner import Learner
import Util

if __name__ == '__main__':
    vocab_dict = Util.get_vocab_dist()
    first_level_dict, second_level_dict, third_level_dict, _, _, _ = Util.get_label_dict()

    workbook = xlrd.open_workbook('dataset/货物分类训练数据.xlsx')
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(1, table.nrows)]
    np.random.shuffle(lines)
    datas = []
    labels1 = []
    labels2 = []
    labels3 = []
    for line in lines:
        cls1 = line[1].value.strip()
        if cls1 not in first_level_dict:
            cls1 = '其它货类'
        labels1.append(first_level_dict[cls1])
        cls2 = line[2].value.strip()
        if cls2 not in second_level_dict:
            cls2 = '其它'
        labels2.append(second_level_dict[cls2])
        cls3 = line[3].value.strip()
        if cls3 not in third_level_dict:
            cls3 = '其它'
        labels3.append(third_level_dict[cls3])
        datas.append(Util.get_word_index(Util.split(line[0].value.strip()), vocab_dict))
    labels1 = np.eye(len(first_level_dict))[labels1]
    labels2 = np.eye(len(second_level_dict))[labels2]
    labels3 = np.eye(len(third_level_dict))[labels3]

    training_num = int(0.8 * len(datas))
    print("training num:", training_num)
    training_datas = datas[: training_num]
    training_labels1 = labels1[: training_num]
    training_labels2 = labels2[: training_num]
    training_labels3 = labels3[: training_num]

    print("val num:", len(labels1) - training_num)
    val_datas = datas[training_num:]
    val_labels1 = labels1[training_num:]
    val_labels2 = labels2[training_num:]
    val_labels3 = labels3[training_num:]

    learner = Learner(training_datas, training_labels1, training_labels2, training_labels3,
                      val_datas, val_labels1, val_labels2, val_labels3,
                      len(first_level_dict), len(second_level_dict), len(third_level_dict), Util.vocab_num + 2, Util.sen_len)
    learner.train(200)
