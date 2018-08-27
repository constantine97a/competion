# -*- coding: utf-8 -*-

# @Time    : 2018/6/26
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Train.py

import xlrd
import numpy as np
from Learner import Learner

if __name__ == '__main__':
    workbook = xlrd.open_workbook('dataset/定价系统训练数据.xlsx')
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(1, table.nrows)]
    np.random.shuffle(lines)
    datas = []
    labels = []
    for line in lines:
        if line[-2].value.strip() != '' and line[5].value.strip() != '':
            label = float(line[table.ncols - 1].value)
            if label > 0 and label < 1000:
                x = [float(c.value) for c in line[1 : 6]]
                x.extend([float(c.value) for c in line[10 : 13]])
                datas.append(x)
                labels.append([label])

    training_num = int(0.85 * len(datas))
    print("training num:", training_num)
    training_datas = datas[ : training_num]
    training_labels = labels[ : training_num]

    training_mean = np.mean(training_datas, axis = 0)
    training_std = np.std(training_datas, axis = 0)
    training_datas = (training_datas - np.tile(training_mean, (len(training_datas), 1)))\
            / np.tile(training_std, (len(training_datas), 1))

    fr = open('logs/norm_params.txt', 'a')
    for i in range(8):
        fr.write(str(training_mean[i]) + ' ')
    fr.write('\n')
    for i in range(8):
        fr.write(str(training_std[i]) + ' ')
    fr.close()

    print("val num:", len(labels) - training_num)
    val_datas = datas[training_num : ]
    val_labels = labels[training_num : ]

    val_datas = (val_datas - np.tile(training_mean, (len(val_datas), 1))) \
                     / np.tile(training_std, (len(val_datas), 1))

    learner = Learner(training_datas, training_labels, val_datas, val_labels, 8)
    learner.train(600)
