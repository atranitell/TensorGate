# -*- coding: utf-8 -*-
""" regression task for kinface
    updated: 2017/05/13
"""

import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework
import tensorflow.contrib.layers as layers

import gate
from issue.image import kinface_utils


def get_loss(end_points_1, logits_1, end_points_2, logits_2,
             labels, num_classes, batch_size):
    with tf.name_scope('loss'):
        x = end_points_1['fc4']
        # x = x / tf.reshape(tf.norm(x, axis=1), [batch_size, 1])
        y = end_points_2['fc4']
        # y = y / tf.reshape(tf.norm(y, axis=1), [batch_size, 1])

        net = tf.concat([x, y], axis=1)
        logits = layers.fully_connected(
            net, num_classes,
            biases_initializer=tf.zeros_initializer(),
            weights_initializer=tf.truncated_normal_initializer(
                stddev=1.0 / 192 * 2),
            weights_regularizer=None,
            activation_fn=None,
            scope='logits')

        logits = tf.to_float(tf.reshape(logits, [batch_size, num_classes]))

        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        losses = tf.reduce_mean(loss1)

        return losses, logits


def get_error(logits, labels, batch_size, num_classes):
    with tf.name_scope('error'):
        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        error = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        return error, predictions


def train(data_name, net_name, chkp_path=None, exclusions=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(data_name, 'train')

            # reset_training_path
            if chkp_path is not None:
                os.rmdir(dataset.log.train_dir)
                dataset.log.train_dir = chkp_path

            # get data
            images_1, images_2, labels, _, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()
            logits_1, end_points_1 = gate.net.factory.get_network(
                net_name, 'train', images_1, dataset.num_classes, '1')
            logits_2, end_points_2 = gate.net.factory.get_network(
                net_name, 'train', images_2, dataset.num_classes, '2')

        losses, logits = get_loss(
            end_points_1, logits_1, end_points_2, logits_2,
            labels, dataset.num_classes, dataset.batch_size)

        error, predictions = get_error(
            logits, labels, dataset.batch_size, dataset.num_classes)

        with tf.name_scope('train'):
            # iter must be the first scalar
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('loss', losses)
            tf.summary.scalar('error', error)

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        with tf.device(dataset.device):
            updater = gate.solver.Updater()
            updater.init_default_updater(
                dataset, global_step, losses, exclusions)

            learning_rate = updater.get_learning_rate()
            restore_saver = updater.get_variables_saver()
            train_op = updater.get_train_op()

        # -------------------------------------------
        # Check point
        # -------------------------------------------
        with tf.name_scope('checkpoint'):
            snapshot = gate.solver.Snapshot()
            chkp_hook = snapshot.get_chkp_hook(dataset)
            summary_hook = snapshot.get_summary_hook(dataset)
            summary_test = snapshot.get_summary_test(dataset)

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_loss, self.duration = 0, 0
                self.mean_err = 0.0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, error, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_loss += run_values.results[1]
                self.mean_err += run_values.results[2]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _loss = self.mean_loss / _invl
                    _err = self.mean_err / _invl
                    _lr = str(run_values.results[3])
                    _duration = self.duration * 1000 / _invl
                    # there time is the running time of a iteration
                    # (if 1 GPU, it is a batch)
                    format_str = 'Iter:%d, loss:%.4f, lr:%s, err:%.4f, time:%.2fms.'
                    gate.utils.show.TRAIN(
                        format_str % (cur_iter, _loss, _lr, _err, _duration))
                    # set zero
                    self.mean_loss, self.mean_err, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    # test_start_time = time.time()
                    test(data_name, net_name, dataset.log.train_dir, summary_test)
                    # test_duration = time.time() - test_start_time

        # record running information
        running_hook = Running_Hook()

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        _iter = 0
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook,
                       tf.train.NanTensorHook(losses)],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, restore_saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
                # _iter += 1
                # if _iter % 20 == 0:
                #     print(mon_sess.run([logits, labels, predictions]))


def test(name, net_name, chkp_path=None, summary_writer=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'test')

            if summary_writer is not None:
                dataset.log.test_dir = chkp_path + '/test/'
            else:
                dataset.log.test_dir = chkp_path + '/val/'

            if not os.path.exists(dataset.log.test_dir):
                os.mkdir(dataset.log.test_dir)

            images_1, images_2, labels, filenames1, filenames2 = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits_1, end_points_1 = gate.net.factory.get_network(
                net_name, 'train', images_1, dataset.num_classes, '1')
            logits_2, end_points_2 = gate.net.factory.get_network(
                net_name, 'train', images_2, dataset.num_classes, '2')

        losses, logits = get_loss(
            end_points_1, logits_1, end_points_2, logits_2,
            labels, dataset.num_classes, dataset.batch_size)

        error, predictions = get_error(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # -------------------------------------------
        # restore from checkpoint
        # -------------------------------------------
        saver = tf.train.Saver(name='restore_all')
        with tf.Session() as sess:
            # load checkpoint
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # start queue from runner
            coord = tf.train.Coordinator()
            threads = []
            for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queuerunner.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # Initial some variables
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            predictions_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            test_batch_info = filenames1 + tab + filenames2 + \
                tab + labels_str + tab + predictions_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')
            gate.utils.show.TEST('Output file in %s.' % test_info_path)

            # progressive bar
            progress_bar = gate.utils.Progressive(min_scale=2.0)

            # -------------------------------------------
            # Start to TEST
            # -------------------------------------------
            err, loss = 0, 0
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _err, _info = sess.run(
                    [losses, error, test_batch_info])
                loss += _loss
                err += _err
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage
                progress_bar.add_float(cur, num_iter)

            test_info_fp.close()

            loss = 1.0 * loss / num_iter
            err = 1.0 * err / num_iter

            # -------------------------------------------
            # output
            # -------------------------------------------
            print()
            gate.utils.show.TEST(
                'Iter:%d, total val sample:%d, num_batch:%d, loss:%.4f, err:%.4f' %
                (int(global_step), dataset.total_num, num_iter, loss, err))

            # -------------------------------------------
            # Summary
            # -------------------------------------------
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/loss', simple_value=loss)
                summary.value.add(tag='test/error', simple_value=err)
                summary_writer.add_summary(summary, global_step)

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return err
