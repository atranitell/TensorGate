# -*- coding: utf-8 -*-
""" regression task for image
    updated: 2017/3/25
"""

import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

import gate


def get_loss(end_points, logits, labels, num_classes, batch_size):
    with tf.name_scope('loss'):
        logits = tf.to_float(tf.reshape(logits, [batch_size, 1]))
        _labels = tf.to_float(tf.reshape(labels, [batch_size, 1]))
        _labels = tf.divide(_labels, num_classes)
        losses = tf.nn.l2_loss([_labels - logits], name='l2_loss')
        return logits, _labels, losses


def get_error(logits, labels, num_classes):
    with tf.name_scope('error'):
        err_mae = tf.reduce_mean(
            input_tensor=tf.abs((logits - labels) * num_classes), name='err_mae')
        err_mse = tf.reduce_mean(
            input_tensor=tf.square((logits - labels) * num_classes), name='err_mse')
        return err_mae, err_mse


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

            # ouput information
            gate.utils.info.print_basic_information(dataset, net_name)

            # get data
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        # with tf.device(dataset.device):
        global_step = framework.create_global_step()

        logits, end_points = gate.net.factory.get_network(
            net_name, 'train', images, 1)

        logits, labels, losses = get_loss(end_points, logits, labels,
                                          dataset.num_classes, dataset.batch_size)

        err_mae, err_mse = get_error(logits, labels, dataset.num_classes)

        with tf.name_scope('train'):
            # iter must be the first scalar
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('err_mae', err_mae)
            tf.summary.scalar('err_mse', err_mse)
            tf.summary.scalar('loss', losses)

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        updater = gate.solver.Updater()
        updater.init_default_updater(dataset, global_step, losses, exclusions)

        learning_rate = updater.get_learning_rate()
        saver = updater.get_variables_saver()
        train_op = updater.get_train_op()

        # -------------------------------------------
        # Check point
        # -------------------------------------------
        snapshot = gate.solver.Snapshot()
        chkp_hook = snapshot.get_chkp_hook(dataset)
        summary_hook = snapshot.get_summary_hook(dataset)
        summary_test = snapshot.get_summary_test(dataset)

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0
                self.best_iter_mae, self.best_mae = 0.0, 1000.0
                self.best_iter_rmse, self.best_rmse = 0.0, 1000.0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, err_mae, err_mse, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_loss += run_values.results[1]
                self.mean_mae += run_values.results[2]
                self.mean_mse += run_values.results[3]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _loss = self.mean_loss / _invl
                    _mae = self.mean_mae / _invl
                    _rmse = math.sqrt(self.mean_mse / _invl)
                    _lr = str(run_values.results[4])
                    _duration = self.duration * 1000 / _invl
                    # there time is the running time of a iteration
                    # (if 1 GPU, it is a batch)
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, mae:%.4f, rmse:%.4f, '
                    format_str += 'lr:%s, time:%.2fms.'
                    print(format_str %
                          (cur_iter, _loss, _mae, _rmse, _lr, _duration))
                    # set zero
                    self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_mae, test_rmse = test(data_name, net_name,
                                               dataset.log.train_dir, summary_test)
                    test_duration = time.time() - test_start_time

                    if test_mae < self.best_mae:
                        self.best_mae = test_mae
                        self.best_iter_mae = cur_iter
                    if test_rmse < self.best_rmse:
                        self.best_rmse = test_rmse
                        self.best_iter_rmse = cur_iter

                    print('[TEST] Test Time: %fs, best MAE: %f in %d, best RMSE: %f in %d.' %
                          (test_duration, self.best_mae, self.best_iter_mae,
                           self.best_rmse, self.best_iter_rmse))

        # record running information
        running_hook = Running_Hook()

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook,
                       tf.train.NanTensorHook(losses)],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def test(name, net_name, chkp_path=None, summary_writer=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        dataset = gate.dataset.factory.get_dataset(name, 'test')
        dataset.log.test_dir = chkp_path + '/test/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        gate.utils.info.print_basic_information(dataset, net_name)

        images, labels_orig, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        logits, end_points = gate.net.factory.get_network(
            net_name, 'test', images, 1)

        logits, labels, losses = get_loss(end_points, logits, labels_orig,
                                          dataset.num_classes, dataset.batch_size)

        err_mae, err_mse = get_error(logits, labels, dataset.num_classes)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # -------------------------------------------
            # restore from checkpoint
            # -------------------------------------------
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            # -------------------------------------------
            # start queue from runner
            # -------------------------------------------
            coord = tf.train.Coordinator()
            threads = []
            for queuerunner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(queuerunner.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            # -------------------------------------------
            # Initial some variables
            # -------------------------------------------
            num_iter = int(math.ceil(dataset.total_num / dataset.batch_size))
            mae, rmse, loss = 0, 0, 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels_orig, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')
            print('[TEST] Output file in %s.' % test_info_path)

            # progressive bar
            progress_bar = gate.utils.Progressive(min_scale=2.0)

            # -------------------------------------------
            # Start to TEST
            # -------------------------------------------
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _mae, _rmse, _info = sess.run(
                    [losses, err_mae, err_mse, test_batch_info])
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
            print('[TEST] Loss:%.4f, mae:%.4f, rmse:%.4f' % (loss, mae, rmse))

            # -------------------------------------------
            # Especially for avec2014
            # -------------------------------------------
            if name == 'avec2014' or name == 'avec2014_flow':
                mae, rmse = gate.dataset.dataset_avec2014_utils.get_accurate_from_file(
                    test_info_path)
                print('[TEST] Loss:%.4f, video_mae:%.4f, video_rmse:%.4f' %
                      (loss, mae, rmse))

            # -------------------------------------------
            # Summary
            # -------------------------------------------
            summary = tf.Summary()
            summary.value.add(tag='test/iter', simple_value=int(global_step))
            summary.value.add(tag='test/mae', simple_value=mae)
            summary.value.add(tag='test/rmse', simple_value=rmse)
            summary.value.add(tag='test/loss', simple_value=loss)
            summary_writer.add_summary(summary, global_step)

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return mae, rmse
