# -*- coding: utf-8 -*-
""" classification task for image
    updated: 2017/05/19
"""
import os
import math
import time

import tensorflow as tf
from tensorflow.contrib import framework

import gate
from gate.utils.logger import logger


def get_network(images, dataset, phase):
    # get Deep Neural Network
    logits, end_points = gate.net.factory.get_network(
        dataset.hps, phase, images, dataset.num_classes)
    return logits, end_points


def get_loss(logits, labels, batch_size, num_classes):
    # get loss
    losses, _labels, _logits = gate.loss.softmax.get_loss(
        logits, labels, num_classes, batch_size)
    # get error
    err, predictions = gate.loss.softmax.get_error(
        _logits, _labels)
    return losses, predictions, err


def train(data_name, chkp_path=None, exclusions=None):
    """ train cnn model
    Args:
        data_name:
        chkp_path:
        exclusion:
    Return:
        None
    """
    with tf.Graph().as_default():
        # get data model
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'train', chkp_path)

        # build data model
        images, labels, _ = dataset.loads()

        # get global step
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # get network
        logits, nets = get_network(images, dataset, 'train')

        # get loss
        losses, logits, err = get_loss(logits, labels, dataset.batch_size,
                                       dataset.num_classes)

        # get updater
        with tf.name_scope('updater'):
            updater = gate.solver.Updater()
            updater.init_default_updater(dataset, global_step,
                                         losses, exclusions)
            learning_rate = updater.get_learning_rate()
            restore_saver = updater.get_variables_saver()
            train_op = updater.get_train_op()

        # Check point
        with tf.name_scope('checkpoint'):
            snapshot = gate.solver.Snapshot()
            chkp_hook = snapshot.get_chkp_hook(dataset)
            summary_hook = snapshot.get_summary_hook(dataset)
            summary_test = snapshot.get_summary_test(dataset)

        # Running Info
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_loss, self.duration, self.mean_err = 0, 0, 0
                self.best_iter_err, self.best_err = 0, 100

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, err, learning_rate],
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

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('loss', _loss, float)
                    f_str.add('err', _err, float)
                    f_str.add('lr', _lr)
                    f_str.add('time', _duration, float)
                    logger.train(f_str.get())

                    # set zero
                    self.mean_err, self.mean_loss, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_err = test(data_name, dataset.log.train_dir,
                                    summary_test)
                    test_duration = time.time() - test_start_time

                    if test_err < self.best_err:
                        self.best_err = test_err
                        self.best_iter_err = cur_iter

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('err', self.best_err, float)
                    f_str.add('in', self.best_iter_err, int)
                    f_str.add('time', test_duration, float)
                    logger.test(f_str.get())

        # record running information
        running_hook = Running_Hook()

        # Start to train
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook,
                       tf.train.NanTensorHook(losses)],
                config=tf.ConfigProto(allow_soft_placement=True),
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, restore_saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def test(data_name, chkp_path, summary_writer=None):
    """ test for regression net
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'test', chkp_path)

        # load data
        images, labels, filenames = dataset.loads()

        # get network
        logits, nets = get_network(images, dataset, 'test')

        # get loss
        losses, logits, err = get_loss(logits, labels, dataset.batch_size,
                                       dataset.num_classes)

        # get saver
        saver = tf.train.Saver(name='restore_all_test')

        with tf.Session() as sess:
            # get latest checkpoint
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
            mean_err, mean_loss = 0, 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for _ in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, err, test_batch_info]
                _loss, _err, _info = sess.run(feeds)
                mean_loss += _loss
                mean_err += _err

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter
            mean_err = 1.0 * mean_err / num_iter

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total_sample', dataset.total_num, int)
            f_str.add('num_batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('err', mean_err, float)
            logger.test(f_str.get())

            # write to summary
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/err', simple_value=mean_err)
                summary.value.add(tag='test/loss', simple_value=mean_loss)
                summary_writer.add_summary(summary, global_step)

            return mean_err
