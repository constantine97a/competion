# -*- coding: utf-8 -*-

# @Time    : 2018/8/13
# @Author  : GIFTED-BOY
# @Email   : 120138380@qq.com
# @File    : Learner.py

import tensorflow as tf
import numpy as np

class Learner(object):
    def __init__(self, training_datas, training_labels1, training_labels2, training_labels3,
                 val_datas, val_labels1, val_labels2, val_labels3, cls_num1, cls_num2, cls_num3,
                 vocab_num, sen_len, embedding_size = 16, filter_sizes = [1, 2, 4], filter_num = 4,
                 batch_size = 256, alpha = 0.01, decay_rate = 0.95):
        self.training_datas = training_datas
        self.training_labels1 = training_labels1
        self.training_labels2 = training_labels2
        self.training_labels3 = training_labels3
        self.val_datas = val_datas
        self.val_labels1 = val_labels1
        self.val_labels2 = val_labels2
        self.val_labels3 = val_labels3
        self.cls_num1 = cls_num1
        self.cls_num2 = cls_num2
        self.cls_num3 = cls_num3
        self.vocab_num = vocab_num
        self.sen_len = sen_len
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num
        self.batch_size = batch_size
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.init_network_params()
        self.create_network()

    def init_network_params(self):
        self.X = tf.placeholder(tf.int32, [None, self.sen_len], name = 'X')
        self.Y1 = tf.placeholder(tf.float32, [None, self.cls_num1], name = 'Y1')
        self.Y2 = tf.placeholder(tf.float32, [None, self.cls_num2], name = 'Y2')
        self.Y3 = tf.placeholder(tf.float32, [None, self.cls_num3], name = 'Y3')
        self.word_embedding = tf.Variable(tf.random_uniform([self.vocab_num, self.embedding_size], -1.0, 1.0), name = 'word_embedding')

        self.conv_A_w = []
        self.conv_A_b = []
        self.conv_B_w = []
        self.conv_B_b = []
        for i in range(len(self.filter_sizes)):
            self.conv_A_w.append(tf.Variable(tf.truncated_normal([self.filter_sizes[i], self.embedding_size, 1, self.filter_num], stddev = 0.1)))
            self.conv_A_b.append(tf.Variable(tf.constant(0.1, shape = [self.filter_num])))
            self.conv_B_w.append(tf.Variable(tf.truncated_normal([self.filter_sizes[i], self.embedding_size, 1, self.filter_num], stddev = 0.1)))
            self.conv_B_b.append(tf.Variable(tf.constant(0.1, shape = [self.filter_num, self.embedding_size])))

        self.conv_output_size_A = self.filter_num * len(self.filter_sizes)
        self.conv_output_size_B = self.embedding_size * self.filter_num * len(self.filter_sizes)
        fc_input_size = self.conv_output_size_A + self.conv_output_size_B
        print(fc_input_size)

        first_level_fc_unit_num1 = 256
        self.first_level_fc_w1 = tf.Variable(tf.truncated_normal([fc_input_size, first_level_fc_unit_num1], stddev = 0.1))
        self.first_level_fc_b1 = tf.Variable(tf.constant(0.1, shape = [first_level_fc_unit_num1]))

        first_level_fc_unit_num2 = 64
        self.first_level_fc_w2 = tf.Variable(tf.truncated_normal([first_level_fc_unit_num1, first_level_fc_unit_num2], stddev = 0.1))
        self.first_level_fc_b2 = tf.Variable(tf.constant(0.1, shape = [first_level_fc_unit_num2]))

        self.first_level_output_w = tf.Variable(tf.truncated_normal([first_level_fc_unit_num2, self.cls_num1], stddev = 0.1))
        self.first_level_output_b = tf.Variable(tf.constant(0.1, shape = [self.cls_num1]))


        second_level_fc_unit_num1 = 256
        self.second_level_fc_w1 = tf.Variable(tf.truncated_normal([fc_input_size + 1, second_level_fc_unit_num1], stddev = 0.1))
        self.second_level_fc_b1 = tf.Variable(tf.constant(0.1, shape = [second_level_fc_unit_num1]))

        second_level_fc_unit_num2 = 64
        self.second_level_fc_w2 = tf.Variable(
            tf.truncated_normal([second_level_fc_unit_num1, second_level_fc_unit_num2], stddev = 0.1))
        self.second_level_fc_b2 = tf.Variable(tf.constant(0.1, shape = [second_level_fc_unit_num2]))

        self.second_level_output_w = tf.Variable(
            tf.truncated_normal([second_level_fc_unit_num2, self.cls_num2], stddev = 0.1))
        self.second_level_output_b = tf.Variable(tf.constant(0.1, shape = [self.cls_num2]))


        third_level_fc_unit_num1 = 256
        self.third_level_fc_w1 = tf.Variable(
            tf.truncated_normal([fc_input_size + 2, third_level_fc_unit_num1], stddev = 0.1))
        self.third_level_fc_b1 = tf.Variable(tf.constant(0.1, shape = [third_level_fc_unit_num1]))

        third_level_fc_unit_num2 = 64
        self.third_level_fc_w2 = tf.Variable(
            tf.truncated_normal([third_level_fc_unit_num1, third_level_fc_unit_num2], stddev = 0.1))
        self.third_level_fc_b2 = tf.Variable(tf.constant(0.1, shape = [third_level_fc_unit_num2]))

        self.third_level_output_w = tf.Variable(
            tf.truncated_normal([third_level_fc_unit_num2, self.cls_num3], stddev = 0.1))
        self.third_level_output_b = tf.Variable(tf.constant(0.1, shape=  [self.cls_num3]))


        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        self.global_step = tf.Variable(0, trainable = False)
        self.learning_rate = tf.train.exponential_decay(self.alpha, self.global_step,
                                                        decay_steps = len(self.training_datas) / self.batch_size,
                                                        decay_rate = self.decay_rate)

    def get_embedding_X(self, X):
        X_embedding = tf.nn.embedding_lookup(self.word_embedding, X)
        return tf.expand_dims(X_embedding, -1)

    def create_conv_layerA(self, X):
        conv_output = []
        for i in range(len(self.filter_sizes)):
            conv = tf.nn.relu(tf.nn.conv2d(X, self.conv_A_w[i], strides = [1, 1, 1, 1], padding = 'VALID') + self.conv_A_b[i])
            max_pool = tf.reduce_max(conv, axis = 1)
            conv_output.append(max_pool)
        return tf.reshape(tf.concat(conv_output, 2), [-1, self.conv_output_size_A])

    def create_conv_layerB(self, X):
        X_split = tf.unstack(X, axis = 2)
        conv_output = []
        for i in range(len(self.filter_sizes)):
            w_split = tf.unstack(self.conv_B_w[i], axis = 1)
            b_split = tf.unstack(self.conv_B_b[i], axis = 1)
            convs = []
            for j in range(self.embedding_size):
                convs.append(tf.nn.relu(tf.nn.conv1d(X_split[j], w_split[j], stride = 1, padding = 'VALID') + b_split[j]))
            max_pool = tf.reduce_max(tf.stack(convs, axis = 2), axis = 1)
            conv_output.append(max_pool)
        return tf.reshape(tf.concat(conv_output, axis = 2), [-1, self.conv_output_size_B])

    def create_fc_layer(self):
        conv_output_A = self.create_conv_layerA(self.get_embedding_X(self.X))
        conv_output_B = self.create_conv_layerB(self.get_embedding_X(self.X))
        conv_output = tf.concat([conv_output_A, conv_output_B], axis = 1, name = 'conv_output')

        first_level_fc_output1 = tf.nn.relu(tf.matmul(conv_output, self.first_level_fc_w1) + self.first_level_fc_b1)
        first_level_fc_output_dropout1 = tf.nn.dropout(first_level_fc_output1, self.keep_prob)
        first_level_fc_output2 = tf.nn.relu(tf.matmul(first_level_fc_output_dropout1, self.first_level_fc_w2) + self.first_level_fc_b2)
        first_level_fc_output_dropout2 = tf.nn.dropout(first_level_fc_output2, self.keep_prob)
        self.first_level_output = tf.nn.softmax(
            tf.matmul(first_level_fc_output_dropout2, self.first_level_output_w) + self.first_level_output_b, name = 'first_level_output')

        second_level_fc_input = tf.concat(
            [conv_output, tf.reshape(tf.cast(tf.argmax(self.first_level_output, axis = 1), dtype = tf.float32), [-1, 1])], axis = 1)
        second_level_fc_output1 = tf.nn.relu(tf.matmul(second_level_fc_input, self.second_level_fc_w1) + self.second_level_fc_b1)
        second_level_fc_output_dropout1 = tf.nn.dropout(second_level_fc_output1, self.keep_prob)
        second_level_fc_output2 = tf.nn.relu(
            tf.matmul(second_level_fc_output_dropout1, self.second_level_fc_w2) + self.second_level_fc_b2)
        second_level_fc_output_dropout2 = tf.nn.dropout(second_level_fc_output2, self.keep_prob)
        self.second_level_output = tf.nn.softmax(
            tf.matmul(second_level_fc_output_dropout2, self.second_level_output_w) + self.second_level_output_b, name = 'second_level_output')

        third_level_fc_input = tf.concat(
            [conv_output, tf.reshape(tf.cast(tf.argmax(self.first_level_output, axis = 1), dtype = tf.float32), [-1, 1])], axis = 1)
        third_level_fc_input = tf.concat(
            [third_level_fc_input, tf.reshape(tf.cast(tf.argmax(self.second_level_output, axis = 1), dtype = tf.float32), [-1, 1])], axis = 1)
        third_level_fc_output1 = tf.nn.relu(
            tf.matmul(third_level_fc_input, self.third_level_fc_w1) + self.third_level_fc_b1)
        third_level_fc_output_dropout1 = tf.nn.dropout(third_level_fc_output1, self.keep_prob)
        third_level_fc_output2 = tf.nn.relu(
            tf.matmul(third_level_fc_output_dropout1, self.third_level_fc_w2) + self.third_level_fc_b2)
        third_level_fc_output_dropout2 = tf.nn.dropout(third_level_fc_output2, self.keep_prob)
        self.third_level_output = tf.nn.softmax(
            tf.matmul(third_level_fc_output_dropout2, self.third_level_output_w) + self.third_level_output_b, name = 'third_level_output')

    def create_network(self):
        self.create_fc_layer()
        self.cross_entropy1 = -tf.reduce_sum(self.Y1 * tf.log(self.first_level_output + 1e-10), name = 'loss1')
        self.cross_entropy2 = -tf.reduce_sum(self.Y2 * tf.log(self.second_level_output + 1e-10), name = 'loss2')
        self.cross_entropy3 = -tf.reduce_sum(self.Y3 * tf.log(self.third_level_output + 1e-10), name = 'loss3')
        self.cross_entropy = tf.add(tf.add(self.cross_entropy1, self.cross_entropy2), self.cross_entropy3, name = 'loss')
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy, global_step = self.global_step)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state("models/")
        # saver.restore(self.session, ckpt.model_checkpoint_path)

    def train(self, epoches):
        print("batch size:", self.batch_size)
        batch_num = int(np.ceil(len(self.training_datas) / self.batch_size))
        print("training...")
        saver = tf.train.Saver()
        for i in range(epoches):
            index_array = list(range(batch_num))
            total_loss1 = 0.0
            total_loss2 = 0.0
            total_loss3 = 0.0
            total_loss = 0.0
            for j in range(batch_num):
                batch_index = np.random.choice(index_array)
                left = int(batch_index * self.batch_size)
                right = min(left + self.batch_size, len(self.training_datas))
                batch_x = self.training_datas[left: right]
                batch_y1 = self.training_labels1[left: right]
                batch_y2 = self.training_labels2[left: right]
                batch_y3 = self.training_labels3[left: right]
                self.global_step = i
                _, loss1, loss2, loss3, loss = self.session.run(
                    [self.train_op, self.cross_entropy1, self.cross_entropy2, self.cross_entropy3, self.cross_entropy],
                    feed_dict = {self.X: batch_x, self.Y1: batch_y1, self.Y2: batch_y2, self.Y3: batch_y3, self.keep_prob: 1.0})
                total_loss1 += loss1
                total_loss2 += loss2
                total_loss3 += loss3
                total_loss += loss
                index_array = np.delete(index_array, list(index_array).index(batch_index))
            print("\tepoch", i)
            print("\t\tavg loss1:", total_loss1 / batch_num)
            print("\t\tavg loss2:", total_loss2 / batch_num)
            print("\t\tavg loss3:", total_loss3 / batch_num)
            print("\t\tavg loss:", total_loss / batch_num)
            if (i + 1) % 50 == 0:
                saver.save(self.session, "models/model.ckpt", global_step = i + 1)
                self.test()

    def test(self):
        self.val(self.training_datas, self.training_labels1, self.training_labels2, self.training_labels3,
                 "logs1/loss.log", "logs2/loss.log", "logs3/loss.log", "logs/loss.log",
                 "logs1/acc.log", "logs2/acc.log", "logs3/acc.log", "logs/acc.log")
        self.val(self.val_datas, self.val_labels1, self.val_labels2, self.val_labels3,
                 "logs1/val_loss.log", "logs2/val_loss.log", "logs3/val_loss.log", "logs/val_loss.log",
                 "logs1/val_acc.log", "logs2/val_acc.log", "logs3/val_acc.log", "logs/val_acc.log")

    def val(self, testing_datas, testing_labels1, testing_labels2, testing_labels3,
            loss1_file_name, loss2_file_name, loss3_file_name, loss_file_name,
            acc1_file_name, acc2_file_name, acc3_file_name, acc_file_name):
        fw_loss1 = open(loss1_file_name, "a")
        fw_loss2 = open(loss2_file_name, "a")
        fw_loss3 = open(loss3_file_name, "a")
        fw_loss = open(loss_file_name, "a")
        fw_acc1 = open(acc1_file_name, "a")
        fw_acc2 = open(acc2_file_name, "a")
        fw_acc3 = open(acc3_file_name, "a")
        fw_acc = open(acc_file_name, "a")
        batch_num = int(np.ceil(len(testing_datas) / self.batch_size))
        total_correct_num1 = 0
        total_correct_num2 = 0
        total_correct_num3 = 0
        total_correct_num = 0
        print("num:", len(testing_datas))
        for i in range(batch_num):
            print("batch", i)
            left = i * self.batch_size
            right = min(left + self.batch_size, len(testing_datas))
            batch_x = testing_datas[left : right]
            batch_y1 = testing_labels1[left : right]
            batch_y2 = testing_labels2[left: right]
            batch_y3 = testing_labels3[left: right]
            output_y1, output_y2, output_y3, loss1, loss2, loss3, loss = self.session.run(
                [self.first_level_output, self.second_level_output, self.third_level_output,
                 self.cross_entropy1, self.cross_entropy2, self.cross_entropy3, self.cross_entropy],
                feed_dict = {self.X: batch_x, self.Y1: batch_y1, self.Y2: batch_y2, self.Y3: batch_y3, self.keep_prob: 1.0})
            fw_loss1.write(str(loss1) + "\n")
            fw_loss2.write(str(loss2) + "\n")
            fw_loss3.write(str(loss3) + "\n")
            fw_loss.write(str(loss) + "\n")

            pred_cls1 = np.argmax(output_y1, axis = 1)
            print(pred_cls1)
            real_cls1 = np.argmax(batch_y1, axis = 1)
            correct_num1 = np.sum(pred_cls1 == real_cls1)
            print("\tcorrect num1:", correct_num1)
            total_correct_num1 += correct_num1

            pred_cls2 = np.argmax(output_y2, axis=1)
            print(pred_cls2)
            real_cls2 = np.argmax(batch_y2, axis=1)
            correct_num2 = np.sum(pred_cls2 == real_cls2)
            print("\tcorrect num2:", correct_num2)
            total_correct_num2 += correct_num2

            pred_cls3 = np.argmax(output_y3, axis=1)
            print(pred_cls3)
            real_cls3 = np.argmax(batch_y3, axis=1)
            correct_num3 = np.sum(pred_cls3 == real_cls3)
            print("\tcorrect num3:", correct_num3)
            total_correct_num3 += correct_num3

            correct_num = 0
            for j in range(right - left):
                if pred_cls1[j] == real_cls1[j] and pred_cls2[j] == real_cls2[j] and pred_cls3[j] == real_cls3[j]:
                    correct_num += 1
            print("\tcorrect num:", correct_num)
            total_correct_num += correct_num
        acc1 = total_correct_num1 / len(testing_datas)
        acc2 = total_correct_num2 / len(testing_datas)
        acc3 = total_correct_num3 / len(testing_datas)
        acc = total_correct_num / len(testing_datas)
        fw_acc1.write(str(acc1) + "\n")
        fw_acc2.write(str(acc2) + "\n")
        fw_acc3.write(str(acc3) + "\n")
        fw_acc.write(str(acc) + "\n")
        fw_loss1.close()
        fw_loss2.close()
        fw_loss3.close()
        fw_loss.close()
        fw_acc1.close()
        fw_acc2.close()
        fw_acc3.close()
        fw_acc.close()
