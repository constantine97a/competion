# -*- coding: utf-8 -*-

# @Time    : 2018/8/13
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Test.py

import sys
import xlrd
import Util
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    vocab_dict = Util.get_vocab_dist()
    _, _, _, label_dict1, label_dict2, label_dict3 = Util.get_label_dict()

    workbook = xlrd.open_workbook(sys.argv[1])
    table = workbook.sheet_by_index(0)
    lines = [table.row(i) for i in range(1, table.nrows)]
    datas = []
    for line in lines:
        datas.append(Util.get_word_index(Util.split(line[0].value.strip()), vocab_dict))

    print("test num:", len(datas))
    batch_size = 256
    print("batch size:", batch_size)
    batch_num = int(np.ceil(len(datas) / batch_size))
    print("batch num:", batch_num)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('models/model.ckpt-200.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/'))
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")
        first_level_output = graph.get_tensor_by_name("first_level_output:0")
        second_level_output = graph.get_tensor_by_name("second_level_output:0")
        third_level_output = graph.get_tensor_by_name("third_level_output:0")
        Y1 = []
        Y2 = []
        Y3 = []
        for i in range(batch_num):
            print("\tbatch", i)
            left = i * batch_size
            right = min(left + batch_size, len(datas))
            batch_x = datas[left : right]
            output_y1, output_y2, output_y3 = sess.run([first_level_output, second_level_output, third_level_output],
                                                       feed_dict = {X: batch_x, keep_prob: 1.0})
            pred_cls1 = np.argmax(output_y1, axis = 1)
            pred_cls2 = np.argmax(output_y2, axis = 1)
            pred_cls3 = np.argmax(output_y3, axis = 1)
            Y1.extend(label_dict1[pred_cls1[j]] for j in range(right - left))
            Y2.extend(label_dict2[pred_cls2[j]] for j in range(right - left))
            Y3.extend(label_dict3[pred_cls3[j]] for j in range(right - left))

    fw = open(sys.argv[2], 'w')
    for i in range(len(lines)):
        fw.write(lines[i][0].value.strip() + '\t' + Y1[i] + '\t' + Y2[i] + '\t' + Y3[i] + '\n')
    fw.close()
