# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import os
import math

import tensorflow as tf

from nets import nets_factory
from data import datasets_factory
from util_tools import output


def run(name, net_name, model_path=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        dataset = datasets_factory.get_dataset(name, 'test')
        dataset.log.test_dir = model_path + '/test/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        output.print_basic_information(dataset, net_name)

        images, labels_orig, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        logits, end_points = nets_factory.get_network(
            net_name, 'test', images, 1)

        # ATTENTION!
        logits = tf.to_float(tf.reshape(logits, [dataset.batch_size, 1]))
        labels = tf.to_float(tf.reshape(labels_orig, [dataset.batch_size, 1]))
        labels = tf.div(labels, dataset.num_classes)
        losses = tf.nn.l2_loss([labels - logits], name='l2_loss')

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     tf.train.start_queue_runners(sess=sess)
        #     # print(sess.run(labels.get_shape()))
        #     # print(sess.run(logits.get_shape()))
        #     print(sess.run(logits))

        # raise ValueError(123)

        err_mae = tf.reduce_mean(input_tensor=tf.abs((logits - labels) * dataset.num_classes), name='err_mae')
        err_mse = tf.reduce_mean(input_tensor=tf.square((logits - labels) * dataset.num_classes), name='err_mse')

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # -------------------------------------------
            # restore from checkpoint
            # -------------------------------------------
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('[TEST] Load checkpoint from: %s' % ckpt.model_checkpoint_path)
            else:
                print('[TEST] Non checkpoint file found in %s' % ckpt.model_checkpoint_path)

            # -------------------------------------------
            # start queue from runner
            # -------------------------------------------
            coord = tf.train.Coordinator()
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            # -------------------------------------------
            # Initial some variables
            # -------------------------------------------
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            mae, rmse, loss = 0, 0, 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(labels_orig, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_infp_path = os.path.join(dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_infp_path, 'wb')
            print('[TEST] Output file in %s.' % test_infp_path)

            # progressive bar
            progress_bar = output.progressive(min_scale=2.0)

            # -------------------------------------------
            # Start to TEST
            # -------------------------------------------
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _mae, _rmse, _info = sess.run([losses, err_mae, err_mse, test_batch_info])
                loss += _loss
                mae += _mae
                rmse += _rmse
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage
                progress_bar.add_float(cur, num_iter)

            test_info_fp.close()

            loss = 1.0 * loss / num_iter
            rmse = math.sqrt(1.0 * rmse / num_iter)
            mae = 1.0 * mae / num_iter

            # -------------------------------------------
            # output
            # -------------------------------------------
            print('\n[TEST] Iter:%d, total test sample:%d, num_batch:%d' %
                  (int(global_step), dataset.total_num, num_iter))

            print('[TEST] Loss:%.2f, mae:%.2f, rmse:%.2f' % (loss, mae, rmse))

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
