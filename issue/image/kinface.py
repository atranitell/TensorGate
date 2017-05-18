# -*- coding: utf-8 -*-
""" regression task for kinface
    updated: 2017/05/13
"""

import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

import gate
from issue.image import kinface_utils


def get_loss(end_points_1, logits_1, end_points_2, logits_2,
             labels, num_classes, batch_size):
    with tf.name_scope('loss'):
        labels = tf.to_float(tf.reshape(labels, [batch_size, 1]))

        x = end_points_1['PreLogitsFlatten']
        norm_x = tf.reshape(tf.norm(x, axis=1), [batch_size, 1])
        y = end_points_2['PreLogitsFlatten']
        norm_y = tf.reshape(tf.norm(y, axis=1), [batch_size, 1])

        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 2)
        loss1 = tf.matmul(x[-1], y[-1], transpose_a=True) / (norm_x * norm_y)

        loss1 = tf.reduce_mean(loss1 * labels)
        losses = -loss1 + 1.0
        return losses


def get_error(end_points_1, logits_1, end_points_2, logits_2,
              labels, num_classes, batch_size):
    """ Input batchsize have to be 1.
    """
    with tf.name_scope('loss'):
        labels = tf.to_float(tf.reshape(labels, [batch_size, 1]))

        x = end_points_1['kinface']
        norm_x = tf.reshape(tf.norm(x, axis=1), [batch_size, 1])
        y = end_points_2['kinface']
        norm_y = tf.reshape(tf.norm(y, axis=1), [batch_size, 1])

        loss1 = tf.matmul(x, y, transpose_b=True) / (norm_x * norm_y)
        loss1 = tf.reduce_mean(loss1)
        losses = -loss1 + 1.0
        return losses


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
                net_name, 'train', images_1, 1, '1')
            logits_2, end_points_2 = gate.net.factory.get_network(
                net_name, 'train', images_2, 1, '2')

        with tf.name_scope('loss'):
            losses = get_loss(
                end_points_1, logits_1, end_points_2, logits_2,
                labels, dataset.num_classes, dataset.batch_size)

        with tf.name_scope('train'):
            # iter must be the first scalar
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('loss', losses)

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        with tf.device(dataset.device):
            updater = gate.solver.Updater()
            updater.init_layerwise_updater(
                dataset, global_step, losses, 'net2', 1.0, exclusions)

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
                self.b_iter, self.b_thr, self.b_acc = 0, 0, 0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_loss += run_values.results[1]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _loss = self.mean_loss / _invl
                    _lr = str(run_values.results[2])
                    _duration = self.duration * 1000 / _invl
                    # there time is the running time of a iteration
                    # (if 1 GPU, it is a batch)
                    format_str = 'Iter:%d, loss:%.4f, lr:%s, time:%.2fms.'
                    gate.utils.show.TRAIN(
                        format_str % (cur_iter, _loss, _lr, _duration))
                    # set zero
                    self.mean_loss, self.duration = 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    val_err, threshold = val(
                        data_name, net_name, dataset.log.train_dir, summary_test)

                    test_start_time = time.time()
                    test_err = test(
                        data_name, net_name, threshold, dataset.log.train_dir)
                    test_duration = time.time() - test_start_time

                    if test_err > self.b_acc:
                        self.b_acc = test_err
                        self.b_iter = cur_iter
                        self.b_thr = threshold

                    gate.utils.show.TEST(
                        'Test Time: %.2fs, best ACC: %.4f in %d with threshold %.4f.' %
                        (test_duration, self.b_acc, self.b_iter, self.b_thr))

        # record running information
        running_hook = Running_Hook()

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, restore_saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def val(name, net_name, chkp_path=None, summary_writer=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'val')
            # dataset_test = gate.dataset.factory.get_dataset(name, 'test')

            if summary_writer is not None:
                dataset.log.val_dir = chkp_path + '/val/'
            else:
                dataset.log.val_dir = chkp_path + '/test/'

            if not os.path.exists(dataset.log.val_dir):
                os.mkdir(dataset.log.val_dir)

            images_1, images_2, labels, filenames1, filenames2 = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits_1, end_points_1 = gate.net.factory.get_network(
                net_name, 'train', images_1, 1, '1')
            logits_2, end_points_2 = gate.net.factory.get_network(
                net_name, 'train', images_2, 1, '2')

        with tf.name_scope('loss'):
            losses = get_error(
                end_points_1, logits_1, end_points_2, logits_2,
                labels, dataset.num_classes, dataset.batch_size)

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
            loss = 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            losses_str = tf.as_string(tf.reshape(
                losses, shape=[dataset.batch_size]))
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            test_batch_info = filenames1 + tab + filenames2 + \
                tab + labels_str + tab + losses_str

            test_info_path = os.path.join(
                dataset.log.val_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')
            gate.utils.show.TEST('Output file in %s.' % test_info_path)

            # progressive bar
            progress_bar = gate.utils.Progressive(min_scale=2.0)

            # -------------------------------------------
            # Start to TEST
            # -------------------------------------------
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _info = sess.run(
                    [losses, test_batch_info])
                loss += _loss
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage
                progress_bar.add_float(cur, num_iter)

            test_info_fp.close()

            loss = 1.0 * loss / num_iter

            err, threshold = kinface_utils.get_kinface_error(test_info_path)

            # -------------------------------------------
            # output
            # -------------------------------------------
            print()
            gate.utils.show.TEST(
                'Iter:%d, total val sample:%d, num_batch:%d, loss:%.4f' %
                (int(global_step), dataset.total_num, num_iter, loss))
            gate.utils.show.TEST(
                'Iter:%d, threshold:%.4f, train acc:%.4f' %
                (int(global_step), threshold, err))

            # -------------------------------------------
            # Summary
            # -------------------------------------------
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/loss', simple_value=loss)
                summary_writer.add_summary(summary, global_step)

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return err, threshold


def test(name, net_name, threshold, chkp_path=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'test')
            dataset.log.test_dir = chkp_path + '/test/'

            if not os.path.exists(dataset.log.test_dir):
                os.mkdir(dataset.log.test_dir)

            images_1, images_2, labels, filenames1, filenames2 = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits_1, end_points_1 = gate.net.factory.get_network(
                net_name, 'train', images_1, 1, '1')
            logits_2, end_points_2 = gate.net.factory.get_network(
                net_name, 'train', images_2, 1, '2')

        with tf.name_scope('loss'):
            losses = get_error(
                end_points_1, logits_1, end_points_2, logits_2,
                labels, dataset.num_classes, dataset.batch_size)

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
            loss = 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            losses_str = tf.as_string(tf.reshape(
                losses, shape=[dataset.batch_size]))
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            test_batch_info = filenames1 + tab + filenames2 + \
                tab + labels_str + tab + losses_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')
            gate.utils.show.TEST('Output file in %s.' % test_info_path)

            # progressive bar
            progress_bar = gate.utils.Progressive(min_scale=2.0)

            # -------------------------------------------
            # Start to TEST
            # -------------------------------------------
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _info = sess.run(
                    [losses, test_batch_info])
                loss += _loss
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage
                progress_bar.add_float(cur, num_iter)

            test_info_fp.close()

            loss = 1.0 * loss / num_iter
            err = kinface_utils.get_kinface_error(test_info_path, threshold)

            # -------------------------------------------
            # output
            # -------------------------------------------
            print()
            gate.utils.show.TEST(
                'Iter:%d, total test sample:%d, num_batch:%d, loss:%.4f, acc:%.4f' %
                (int(global_step), dataset.total_num, num_iter, loss, err))

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return err
