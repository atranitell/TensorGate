# -*- coding: utf-8 -*-
""" updated: 2017/3/30
"""
import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

import gate


def run1(is_train, testpath, file, name, net_name, chkp_path):
    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'test')

            dataset.data_path = testpath
            print(dataset.data_path)

            if is_train:
                dataset.total_num = 400

            dataset.log.test_dir = chkp_path + '/test/'
            if not os.path.exists(dataset.log.test_dir):
                os.mkdir(dataset.log.test_dir)

            images, images1, labels_orig, filenames, filenames1 = dataset.loads()
            # images1, images, labels_orig, filenames1, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits, end_points = gate.net.factory.get_network(
                net_name, 'test', images, 128)

        with tf.name_scope('loss'):
            logits = tf.to_float(tf.reshape(
                logits, [dataset.batch_size, 128]))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_orig, logits=logits)
            loss = tf.reduce_mean(losses)

        with tf.name_scope('error'):
            predictions = tf.to_int32(tf.argmax(logits, axis=1))
            err = tf.reduce_mean(tf.to_float(
                tf.equal(predictions, labels_orig)))

        # -------------------------------------------
        # Start to test
        # -------------------------------------------
        saver = tf.train.Saver(name='restore_all')
        with tf.Session() as sess:
            # restore from snapshot
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queue in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            output_err, output_loss = 0, 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels_orig, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            # file info
            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')
            gate.utils.show.TEST('Output file in %s.' % test_info_path)

            # Start to TEST

            features_store = []
            print(end_points['kinface'])
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _err, _info, _feat = sess.run(
                    [loss, err, test_batch_info, end_points['kinface']])
                output_loss += _loss
                output_err += _err
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

                features_store.append(_feat)

            test_info_fp.close()

            import numpy as np
            stack_features = features_store[0]
            # for i in range(1, 2):
            #         stack_features = np.row_stack(
            #             (stack_features, features_store[i]))

            if is_train:
                for i in range(1, 4):
                    stack_features = np.row_stack(
                        (stack_features, features_store[i]))

            actual_features = []
            for i in range(dataset.total_num):
                actual_features.append(stack_features[i])

            print(np.array(actual_features).shape)

            with open(file, 'w') as fw:
                for i in range(dataset.total_num):
                    for j in range(1792):
                        fw.write(str(actual_features[i][j]) + ' ')
                    fw.write('\n')

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def run2(is_train, testpath, file, name, net_name, chkp_path):
    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'test')

            dataset.data_path = testpath
            print(dataset.data_path)

            if is_train:
                dataset.total_num = 400

            dataset.log.test_dir = chkp_path + '/test/'
            if not os.path.exists(dataset.log.test_dir):
                os.mkdir(dataset.log.test_dir)

            # images, images1, labels_orig, filenames, filenames1 = dataset.loads()
            images1, images, labels_orig, filenames1, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits, end_points = gate.net.factory.get_network(
                net_name, 'test', images, 128)

        with tf.name_scope('loss'):
            logits = tf.to_float(tf.reshape(
                logits, [dataset.batch_size, 128]))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_orig, logits=logits)
            loss = tf.reduce_mean(losses)

        with tf.name_scope('error'):
            predictions = tf.to_int32(tf.argmax(logits, axis=1))
            err = tf.reduce_mean(tf.to_float(
                tf.equal(predictions, labels_orig)))

        # -------------------------------------------
        # Start to test
        # -------------------------------------------
        saver = tf.train.Saver(name='restore_all')
        with tf.Session() as sess:
            # restore from snapshot
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queue in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queue.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            output_err, output_loss = 0, 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels_orig, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            # file info
            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')
            gate.utils.show.TEST('Output file in %s.' % test_info_path)

            # Start to TEST
            features_store = []
            print(end_points['kinface'])
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _err, _info, _feat = sess.run(
                    [loss, err, test_batch_info, end_points['kinface']])
                output_loss += _loss
                output_err += _err
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage

                features_store.append(_feat)

            test_info_fp.close()

            import numpy as np
            stack_features = features_store[0]
            # for i in range(1, 2):
            #         stack_features = np.row_stack(
            #             (stack_features, features_store[i]))

            if is_train:
                for i in range(1, 4):
                    stack_features = np.row_stack(
                        (stack_features, features_store[i]))

            actual_features = []
            for i in range(dataset.total_num):
                actual_features.append(stack_features[i])

            print(np.array(actual_features).shape)

            with open(file, 'w') as fw:
                for i in range(dataset.total_num):
                    for j in range(1792):
                        fw.write(str(actual_features[i][j]) + ' ')
                    fw.write('\n')

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

import issue.image.kinface_distance as kinface_distance


def run(name, net_name, chkp_path):
    """ test for classification
    """
    cluster = 'fd_train_'
    is_train = False
    mean_acc = 0.0
    for i in range(1, 6):
        base = cluster + str(i)
        basename = 'issue/image/' + base
        r_fp_name = 'C:/Users/jk/Desktop/Gate/_datasets/kinface2/' + base + '.txt'
        run1(is_train, r_fp_name, basename+'_1.txt', name, net_name, chkp_path)
        run2(is_train, r_fp_name, basename+'_2.txt', name, net_name, chkp_path)

        val = kinface_distance.get_result(
            p_fp=basename + '_1.txt',
            c_fp=basename + '_2.txt',
            r_fp=r_fp_name,
            rw_fp=basename + '_r.txt')
        if is_train is False:
            basename = basename.replace('_train_', '_val_')
            mean_acc += kinface_distance.get_test(basename + '_r.txt', val)

    print('average on val: ', mean_acc / 5.0)
    print()
