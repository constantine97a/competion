# -*- coding: utf-8 -*-

# @Time    : 2018/6/26
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Learner.py

import tensorflow as tf
import numpy as np

class Learner(object):
    def __init__(self, training_datas, training_labels, val_datas, val_labels, features_num,
                 batch_size = 256, alpha = 0.0005, decay_rate = 0.98):
        self.training_datas = training_datas
        self.training_labels = training_labels
        self.val_datas = val_datas
        self.val_labels = val_labels
        self.features_num = features_num
        self.batch_size = batch_size
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.init_params()
        self.create_network()

    def init_params(self):
        self.X = tf.placeholder(dtype = tf.float32, shape = [None, self.features_num], name = 'X')
        self.Y = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = 'Y')

        units_num1 = 32
        self.w1 = tf.Variable(tf.truncated_normal(shape = [self.features_num, units_num1], stddev = 0.1), name = 'w1')
        self.b1 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b1')

        units_num2 = 64
        self.w2 = tf.Variable(tf.truncated_normal(shape = [units_num1, units_num2], stddev = 0.1), name = 'w2')
        self.b2 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b2')

        units_num3 = 128
        self.w3 = tf.Variable(tf.truncated_normal(shape = [units_num2, units_num3], stddev = 0.1), name = 'w3')
        self.b3 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b3')

        units_num4 = 512
        self.w4 = tf.Variable(tf.truncated_normal(shape = [units_num3, units_num4], stddev = 0.1), name = 'w4')
        self.b4 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b4')

        units_num5 = 1024
        self.w5 = tf.Variable(tf.truncated_normal(shape = [units_num4, units_num5], stddev = 0.1), name = 'w5')
        self.b5 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b5')

        units_num6 = 2048
        self.w6 = tf.Variable(tf.truncated_normal(shape = [units_num5, units_num6], stddev = 0.1), name = 'w6')
        self.b6 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b6')

        units_num7 = 2048
        self.w7 = tf.Variable(tf.truncated_normal(shape = [units_num6, units_num7], stddev = 0.1), name = 'w7')
        self.b7 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b7')

        units_num8 = 4096
        self.w8 = tf.Variable(tf.truncated_normal(shape = [units_num7, units_num8], stddev = 0.1), name='w8')
        self.b8 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b8')

        units_num9 = 1024
        self.w9 = tf.Variable(tf.truncated_normal(shape = [units_num8, units_num9], stddev = 0.1), name = 'w9')
        self.b9 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b9')

        units_num10 = 1024
        self.w10 = tf.Variable(tf.truncated_normal(shape = [units_num9, units_num10], stddev = 0.1), name = 'w10')
        self.b10 = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'b10')

        units_num11 = 256
        self.w11 = tf.Variable(tf.truncated_normal(shape=[units_num10, units_num11], stddev=0.1), name='w11')
        self.b11 = tf.Variable(tf.constant(0.1, dtype=tf.float32), name='b11')

        units_num12 = 128
        self.w12 = tf.Variable(tf.truncated_normal(shape=[units_num11, units_num12], stddev=0.1), name='w12')
        self.b12 = tf.Variable(tf.constant(0.1, dtype=tf.float32), name='b12')

        units_num13 = 64
        self.w13 = tf.Variable(tf.truncated_normal(shape=[units_num12, units_num13], stddev=0.1), name='w13')
        self.b13 = tf.Variable(tf.constant(0.1, dtype=tf.float32), name='b13')

        units_num14 = 16
        self.w14 = tf.Variable(tf.truncated_normal(shape=[units_num13, units_num14], stddev=0.1), name='w14')
        self.b14 = tf.Variable(tf.constant(0.1, dtype=tf.float32), name='b14')

        self.output_w = tf.Variable(tf.truncated_normal(shape = [units_num14, 1], stddev = 0.1), name = 'output_w')
        self.output_b = tf.Variable(tf.constant(0.1, dtype = tf.float32), name = 'output_b')

        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate = tf.train.exponential_decay(self.alpha, self.global_step,
                                                        decay_steps = len(self.training_datas) / self.batch_size,
                                                        decay_rate = self.decay_rate)

    def create_network(self):
        a1 = tf.nn.elu(tf.matmul(self.X, self.w1) + self.b1)
        a2 = tf.nn.elu(tf.matmul(a1, self.w2) + self.b2)
        a3 = tf.nn.elu(tf.matmul(a2, self.w3) + self.b3)
        a4 = tf.nn.elu(tf.matmul(a3, self.w4) + self.b4)
        a5 = tf.nn.elu(tf.matmul(a4, self.w5) + self.b5)
        a6 = tf.nn.elu(tf.matmul(a5, self.w6) + self.b6)
        a7 = tf.nn.elu(tf.matmul(a6, self.w7) + self.b7)
        a8 = tf.nn.elu(tf.matmul(a7, self.w8) + self.b8)
        a9 = tf.nn.elu(tf.matmul(a8, self.w9) + self.b9)
        a10 = tf.nn.elu(tf.matmul(a9, self.w10) + self.b10)
        a11 = tf.nn.elu(tf.matmul(a10, self.w11) + self.b11)
        a12 = tf.nn.elu(tf.matmul(a11, self.w12) + self.b12)
        a13 = tf.nn.elu(tf.matmul(a12, self.w13) + self.b13)
        a14 = tf.nn.elu(tf.matmul(a13, self.w14) + self.b14)
        self.output = tf.add(tf.matmul(a14, self.output_w), self.output_b, name = 'output')
        self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.output), name = 'loss')
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state("DL_Models/")
        # saver.restore(self.session, ckpt.model_checkpoint_path)

    def train(self, epoches):
        print("batch size:", self.batch_size)
        batch_num = int(np.ceil(len(self.training_datas) / self.batch_size))
        print("training...")
        saver = tf.train.Saver()
        fw_loss = open("logs/training_loss.log", "a")
        for i in range(epoches):
            index_array = list(range(batch_num))
            total_loss = 0.0
            for j in range(batch_num):
                batch_index = np.random.choice(index_array)
                left = int(batch_index * self.batch_size)
                right = min(left + self.batch_size, len(self.training_datas))
                batch_x = self.training_datas[left : right]
                batch_y = self.training_labels[left : right]
                self.global_step = i
                _, loss = self.session.run([self.train_op, self.loss],
                                           feed_dict = {self.X: batch_x, self.Y: batch_y})
                total_loss += loss
                index_array = np.delete(index_array, list(index_array).index(batch_index))
            print("\tepoch", i, "-----> avg loss:", total_loss / batch_num)
            fw_loss.write(str(total_loss / batch_num) + "\n")
            if (i + 1) % 200 == 0:
                saver.save(self.session, "DL_Models/model.ckpt", global_step = i + 1)
                self.test()
        fw_loss.close()

    def test(self):
        self.val(self.training_datas, self.training_labels, "logs/loss.log", "logs/rmse.log", "logs/out.log")
        self.val(self.val_datas, self.val_labels, "logs/val_loss.log", "logs/val_rmse.log", "logs/val_out.log")

    def val(self, testing_datas, testing_labels, loss_file_name, rmse_file_name, out_file_name):
        fw_loss = open(loss_file_name, "a")
        fw_rmse = open(rmse_file_name, "a")
        fw_out = open(out_file_name, "a")
        batch_num = int(np.ceil(len(testing_datas) / self.batch_size))
        print("val...")
        total_error = 0.0
        total_loss = 0.0
        for i in range(batch_num):
            print("\tbatch", i)
            left = i * self.batch_size
            right = min(left + self.batch_size, len(testing_datas))
            batch_x = testing_datas[left : right]
            batch_y = testing_labels[left : right]
            output_y, loss = self.session.run([self.output, self.loss],
                                              feed_dict = {self.X: batch_x, self.Y: batch_y})
            out = ''
            for j in range(left, right):
                out += str(output_y[left - j][0]) + '\t' + str(batch_y[left - j][0]) + '\n'
            fw_out.write(out)
            total_loss += loss
            total_error += np.sum((output_y - batch_y) ** 2)
        fw_loss.write(str(total_loss / batch_num) + "\n")
        fw_rmse.write(str(np.sqrt(total_error / len(testing_datas))) + "\n")
        fw_loss.close()
        fw_rmse.close()
        fw_out.close()
