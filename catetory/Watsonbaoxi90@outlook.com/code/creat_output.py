#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from generate import generate_batch
from generate import embed
from generate import id2words
from generate import evaluate
from generate import sentence_token
from generate import pad_sentence_batch
import os
from datetime import datetime
from util import Progbar
import sys
from sklearn import metrics
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

#nu = 1
vec, voc = embed()
vocab_size = len(vec)
batch_size = 307
num_units = [50, 60, 30, 100, 150, 120, 80, 70, 50, 80]
max_gradient_norm = 5
learning_rate = 0.01
n_epochs = 10
n_outputs = 1
train_keep_prob = 1
pwd = os.getcwd()

best_loss = np.infty
best_f1 = 0.0
epochs_without_progress = 0
max_epochs_without_progress = 50
checkpoint_path = "./tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
 

def build_net(num_unit):
    with tf.name_scope("placeholder"):
        encode_x1 = tf.placeholder(tf.int32, shape=[batch_size, None])
        encode_x2 = tf.placeholder(tf.int32, shape=[batch_size, None])
        x1 = tf.transpose(encode_x1, (1, 0))
        x2 = tf.transpose(encode_x2, (1, 0))
        x1_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
        x2_sequence_length = tf.placeholder(tf.int32, shape=[batch_size])
        simi_values = tf.placeholder(tf.int32, shape=[batch_size])
        keep_prob = tf.placeholder_with_default(1.0, shape=())

    with tf.name_scope('wmbeddings'):
        embeddings = tf.Variable(vec, trainable=True, name="embeds")
        x1_emb = tf.nn.embedding_lookup(embeddings, x1)
        x2_emb = tf.nn.embedding_lookup(embeddings, x2)
    with tf.name_scope("decode"):
        with tf.variable_scope("x1"):
            x1_f_cell = tf.nn.rnn_cell.GRUCell(num_unit)
            x1_b_cell = tf.nn.rnn_cell.GRUCell(num_unit)
            x1_f_drop = tf.contrib.rnn.DropoutWrapper(
                x1_f_cell, input_keep_prob=keep_prob)
            x1_b_drop = tf.contrib.rnn.DropoutWrapper(
                x1_b_cell, input_keep_prob=keep_prob)
            x1_outputs, x1_state = tf.nn.bidirectional_dynamic_rnn(
                x1_f_drop, x1_b_drop, x1_emb, sequence_length=x1_sequence_length, time_major=True, dtype=tf.float32)
        with tf.variable_scope("x2"):
            x2_f_cell = tf.nn.rnn_cell.GRUCell(num_unit)
            x2_b_cell = tf.nn.rnn_cell.GRUCell(num_unit)
            x2_f_drop = tf.contrib.rnn.DropoutWrapper(
                x2_f_cell, input_keep_prob=keep_prob)
            x2_b_drop = tf.contrib.rnn.DropoutWrapper(
                x2_b_cell, input_keep_prob=keep_prob)
            x2_outputs, x2_state = tf.nn.bidirectional_dynamic_rnn(
                x2_f_drop,x2_b_drop, x2_emb, sequence_length=x2_sequence_length, time_major=True, dtype=tf.float32)
    with tf.name_scope("logits"):
        x1_f_state,x1_b_state = x1_state
        x2_f_state,x2_b_state = x2_state
        x1_state_new = tf.concat([x1_f_state, x1_b_state], axis=-1)
        x2_state_new = tf.concat([x2_f_state, x2_b_state], axis=-1)
        states = tf.concat([x1_state_new, x2_state_new], axis=-1)
        logits = tf.layers.dense(states, n_outputs)
        labels = tf.reshape(simi_values, (batch_size, 1))
        labels = tf.cast(labels, tf.float32)
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(xentropy)
        loss_summary = tf.summary.scalar('log_loss', loss)
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)
        logits_soft = tf.sigmoid(logits)
        pred = tf.cast(tf.round(logits_soft), tf.int32)
        correct = tf.equal(pred, simi_values)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('log_accuracy', accuracy)
    logdir = log_dir("logreg")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    pram = [encode_x1, encode_x2,x1_sequence_length, x2_sequence_length]
    return saver, pred, pram, init


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)




def train():
    global best_f1
    global epochs_without_progress
    source_file = open(train1)
    target_file = open(train2)
    simi_file = open(train_simi)
    encoder_inputs = []
    decoder_inputs = []
    simi = []
    for line in source_file:
        sentence = sentence_token(line)
        encoder_inputs.append(sentence)
    for line in target_file:
        sentence_input = sentence_token(line)
        decoder_inputs.append(sentence_input)
    for line in simi_file:
        simi_input = int(line.strip())
        simi.append(simi_input)
    source_file.close()
    target_file.close()
    simi_file.close()
    dev1_file = open(dev1)
    dev2_file = open(dev2)
    d_simi = open(dev_simi)
    encoder_dev1 = []
    decoder_dev2 = []
    simi_dev = []
    for line in dev1_file:
        sentence = sentence_token(line)
        encoder_dev1.append(sentence)
    for line in dev2_file:
        sentence_input = sentence_token(line)
        decoder_dev2.append(sentence_input)
    for line in d_simi:
        simi_input = int(line.strip())
        simi_dev.append(simi_input)
    source_file.close()
    target_file.close()
    simi_file.close()
    with tf.Session(config=config) as sess:
        if os.path.isfile(checkpoint_epoch_path):
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)
        for epoch in range(n_epochs):
            train_num = 73461
            # train_num = 32
            prog = Progbar(target=1 + int(train_num / batch_size))
            gen = generate_batch(encoder_inputs, decoder_inputs, simi, batch_size, shuffle=True, is_train=True)
            for i in range(int(train_num / batch_size)):
                
                source_batch_pad, target_batch_pad, simi_input_batch, source_seq_length, target_seq_length =next(gen)
                pred_, logits_, loss_, loss_summary_, _ = sess.run([pred, logits, loss, loss_summary, training_op], feed_dict={
                    encode_x1: source_batch_pad, encode_x2: target_batch_pad,
                    simi_values: simi_input_batch, x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length, keep_prob: train_keep_prob})
                y_true = np.array(simi_input_batch)
                precision = metrics.precision_score(y_true, pred_)
                recall = metrics.recall_score(y_true, pred_)
                f1 = metrics.f1_score(y_true, pred_)
                if i % 10 == 0:
                    file_writer.add_summary(loss_summary_, epoch * train_num + i)
                prog.update(i + 1, [("train loss", loss_), ("precision",
                                                            precision), ("recall", recall), ("f1", f1)])
            print("epoch:{}".format(epoch), "epoch_loss:{:.5f}".format(loss_))
            
            if epoch % 1 == 0:
                dev_num = 7770
                # dev_num = 32
                num = int(dev_num / batch_size)
                dev_accuracy = 0.0
                dev_loss = 0.0
                y_true = []
                y_pred = []
                gen = generate_batch(encoder_dev1, decoder_dev2, simi_dev, batch_size, shuffle=True, is_train=True)
                for i in range(num):
                    
                    dev1_pad, dev2_pad, dev_simi_batch, dev1_seq_length, dev2_seq_length =next(gen)
                    pred_dev, loss_dev = sess.run([pred, loss], feed_dict={
                        encode_x1: dev1_pad, encode_x2: dev2_pad,
                        simi_values: dev_simi_batch, x1_sequence_length: dev1_seq_length, x2_sequence_length: dev2_seq_length})
                    y_true = np.append(y_true, dev_simi_batch)
                    y_pred = np.append(y_pred, pred_dev)
                    dev_loss += loss_dev
                y_true = np.array(y_true)
                precision_dev = metrics.precision_score(y_true, y_pred)
                recall_dev = metrics.recall_score(y_true, y_pred)
                f1_dev = metrics.f1_score(y_true, y_pred)
                print("epoch:{}".format(epoch), "\tDev loss:{:.5f}".format(dev_loss / num), "\tprecision_dev:{:.5f}".format(precision_dev),
                      "\trecall_dev:{:.5f}".format(recall_dev), "\tf1_dev:{:.5f}".format(f1_dev))
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))
                if f1_dev > best_f1:
                    saver.save(sess, final_model_path)
                    best_f1 = f1_dev
                else:
                    epochs_without_progress += 5
                    if epochs_without_progress > max_epochs_without_progress:
                        print("Early stopping")
                        break
        #os.remove(checkpoint_epoch_path)


def finaluse(inputpath, inputpath2, outpath, final_model_path, saver, pred, pram):
    items = []
    cl = []
    encode_x1 = pram[0]
    encode_x2 = pram[1]
    x1_sequence_length = pram[2]
    x2_sequence_length = pram[3]
    a = 0
    for line in open(inputpath, 'r'):
        sec = sentence_token(line)
        items.append(sec)
    print('word num:{}'.format(a))
    for line in open(inputpath2, 'r'):
        sec = sentence_token(line)
        cl.append(sec)
    target , target_seq_length = pad_sentence_batch(cl)
    predicts = np.zeros(len(cl), dtype = int)
    with tf.Session() as sess:
        saver.restore(sess, final_model_path)
        i = 0
        for item in items:
            source, source_seq_length = evaluate(
                item, batch_size)
            preds = sess.run(pred, feed_dict={encode_x1: source, encode_x2: target,
                                            x1_sequence_length: source_seq_length, x2_sequence_length: target_seq_length})

            i += 1
            predicts = np.vstack( [predicts , preds.reshape(307)])
            if i % 100 == 0:
                print('times is :{}'.format(i))
        np.save(outpath, predicts)

# tensorboard --logdir=tf_logs

if __name__ == '__main__':
    inputpath = pwd + '/data/pre1.conll'
    inputpath2 = pwd + '/data/class.conll'

    
    #train()
    for i in [9]:
        saver, pred, pram, init = build_net(num_units[i])
        init
        final_model_path = "./model/my_logreg_model_"+str(i)
        outpath = pwd + '/data/pre_out_' + str(i)
        finaluse(inputpath, inputpath2, outpath,final_model_path,saver, pred,pram)
    # finaluse(inputpath,outpath)
