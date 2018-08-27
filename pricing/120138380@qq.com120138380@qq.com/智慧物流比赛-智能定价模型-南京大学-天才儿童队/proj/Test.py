# -*- coding: utf-8 -*-

# @Time    : 2018/7/4
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Test.py

import sys
import xlrd
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    fr = open('params/norm_params.txt')
    norm_means_line = fr.readline()
    norm_std_line = fr.readline()
    fr.close()
    norm_means = [float(s) for s in norm_means_line.strip().split(' ')]
    norm_std = [float(s) for s in norm_std_line.strip().split(' ')]

    print("reading...")
    workbook = xlrd.open_workbook(sys.argv[1])
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(1, table.nrows)]
    datas = []
    for line in lines:
        x = []
        for i in range(1, 6):
            x.append(float(line[i].value) if str(line[i].value).strip() != '' else norm_means[i])
        for i in range(10, 13):
            x.append(float(line[i].value) if str(line[i].value).strip() != '' else norm_means[i - 5])
        datas.append(x)
    datas = (datas - np.tile(norm_means, (len(datas), 1))) / np.tile(norm_std, (len(datas), 1))

    print("test num:", len(datas))
    batch_size = 256
    print("batch size:", batch_size)
    batch_num = int(np.ceil(len(datas) / batch_size))
    print("batch num:", batch_num)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('DL_Models/model.ckpt-400.meta')
        saver.restore(sess, tf.train.latest_checkpoint('DL_Models/'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        output = graph.get_tensor_by_name("output:0")
        Y = []
        for i in range(batch_num):
            print("\tbatch", i)
            left = i * batch_size
            right = min(left + batch_size, len(datas))
            batch_x = datas[left : right]
            output_y = sess.run(output, feed_dict = {X: batch_x})
            Y.extend(output_y[j][0] for j in range(right - left))

    print("y num:", len(Y))
    fw = open(sys.argv[2], 'w')
    fw.write('\n'.join(str(y) for y in Y))
    fw.close()
