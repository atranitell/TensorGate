# -*- coding: utf-8 -*-
""" regression task for image
    updated: 2017/05/19
"""
import os
import math
import time
import re

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework

import gate
from gate.utils.logger import logger

from project.avec2014 import avec2014_error


def get_network(images, dataset, phase, scope=''):
    # get Deep Neural Network
    logits0, end_points0 = gate.net.factory.get_network(
        dataset.hps, phase, images[0], 1, scope + 'net1')
    logits1, end_points1 = gate.net.factory.get_network(
        dataset.hps, phase, images[1], 1, scope + 'net2')
    logits2, end_points2 = gate.net.factory.get_network(
        dataset.hps, phase, images[2], 1, scope + 'net3')
    logits3, end_points3 = gate.net.factory.get_network(
        dataset.hps, phase, images[3], 1, scope + 'net4')

    logits = [logits0, logits1, logits2, logits3]
    nets = [end_points0, end_points1, end_points2, end_points3]

    return logits, nets


def get_loss(logits, labels, batch_size, num_classes):
    # get loss
    losses1, _, logits1 = gate.loss.l2.get_loss(
        logits[0], labels, num_classes, batch_size)
    losses2, _, logits2 = gate.loss.l2.get_loss(
        logits[1], labels, num_classes, batch_size)
    losses3, _, logits3 = gate.loss.l2.get_loss(
        logits[2], labels, num_classes, batch_size)
    losses4, labels, logits4 = gate.loss.l2.get_loss(
        logits[3], labels, num_classes, batch_size)
    # summary loss
    losses = losses1 + losses2 + losses3 + losses4
    # get error
    logits = 0.25 * (logits1 + logits2 + logits3 + logits4)
    mae, rmse = gate.loss.l2.get_error(logits, labels, num_classes)

    return losses, logits, mae, rmse


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
        images0, images1, images2, images3, labels, _ = dataset.loads()
        images = [images0, images1, images2, images3]

        # get global step
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # get network
        logits, nets = get_network(images, dataset, 'train')

        # get loss
        losses, logits, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # get updater
        with tf.name_scope('updater'):
            updater = gate.solver.Updater()
            updater.init_default_updater(
                dataset, global_step, losses, exclusions)
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
                self.mean_loss, self.duration = 0, 0
                self.mean_mae, self.mean_rmse = 0, 0
                self.best_iter_mae, self.best_mae = 0, 1000
                self.best_iter_rmse, self.best_rmse = 0, 1000

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, losses, mae, rmse, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_loss += run_values.results[1]
                self.mean_mae += run_values.results[2]
                self.mean_rmse += run_values.results[3]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _loss = self.mean_loss / _invl
                    _mae = self.mean_mae / _invl
                    _rmse = self.mean_rmse / _invl
                    _lr = str(run_values.results[4])
                    _duration = self.duration * 1000 / _invl

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('loss', _loss, float)
                    f_str.add('mae', _mae, float)
                    f_str.add('rmse', _rmse, float)
                    f_str.add('lr', _lr)
                    f_str.add('time', _duration, float)
                    logger.train(f_str.get())

                    # set zero
                    self.mean_mae, self.mean_rmse = 0, 0
                    self.mean_loss, self.duration = 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_mae, test_rmse = test(
                        data_name, dataset.log.train_dir, summary_test)
                    test_duration = time.time() - test_start_time

                    if test_mae < self.best_mae:
                        self.best_mae = test_mae
                        self.best_iter_mae = cur_iter
                    if test_rmse < self.best_rmse:
                        self.best_rmse = test_rmse
                        self.best_iter_rmse = cur_iter

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('best mae', self.best_mae, float)
                    f_str.add('in', self.best_iter_mae, int)
                    f_str.add('best rmse', self.best_rmse, float)
                    f_str.add('in', self.best_iter_rmse, int)
                    f_str.add('time', test_duration, float)
                    logger.train(f_str.get())

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
        images0, images1, images2, images3, labels, filenames = dataset.loads()
        images = [images0, images1, images2, images3]

        # get network
        logits, nets = get_network(images, dataset, 'test')

        # get loss
        losses, logits, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

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
            mean_mae, mean_rmse, mean_loss = 0, 0, 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for _ in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, mae, rmse, test_batch_info]
                _loss, _mae, _rmse, _info = sess.run(feeds)
                mean_loss += _loss
                mean_mae += _mae
                mean_rmse += _rmse

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter
            mean_rmse = 1.0 * mean_rmse / num_iter
            mean_mae = 1.0 * mean_mae / num_iter

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('mae', mean_mae, float)
            f_str.add('rmse', mean_rmse, float)
            logger.test(f_str.get())

            # for specify dataset
            # it use different compute method for mae/rmse
            # rewrite the mean_x value
            if dataset.name.find('avec2014') == 0:
                mean_mae, mean_rmse = avec2014_error.get_accurate_from_file(
                    test_info_path, 'img')
                f_str = gate.utils.string.format_iter(global_step)
                f_str.add('loss', mean_loss, float)
                f_str.add('video_mae', mean_mae, float)
                f_str.add('video_rmse', mean_rmse, float)
                logger.test(f_str.get())

            # write to summary
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/mae', simple_value=mean_mae)
                summary.value.add(tag='test/rmse', simple_value=mean_rmse)
                summary.value.add(tag='test/loss', simple_value=mean_loss)
                summary_writer.add_summary(summary, global_step)

            return mean_mae, mean_rmse


def heatmap(name, chkp_path):
    """ generate heatmap
    """
    with tf.Graph().as_default():
        # Preparing the dataset
        dataset = gate.dataset.factory.get_dataset(name, 'test', chkp_path)
        dataset.log.test_dir = chkp_path + '/test_heatmap/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        # load data
        images0, images1, images2, images3, labels, filenames = dataset.loads()
        images = [images0, images1, images2, images3]

        # get network
        logits, nets = get_network(images, dataset, 'test', '')

        # get loss
        losses, logits, mae, rmse = get_loss(
            logits, labels, dataset.batch_size, dataset.num_classes)

        # restore from checkpoint
        saver = tf.train.Saver(name='restore_all')

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
            mean_mae, mean_rmse, mean_loss = 0, 0, 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            # -------------------------------------------------
            # heatmap related
            heatmap_path = chkp_path + '/heatmap/'
            if not os.path.exists(heatmap_path):
                os.mkdir(heatmap_path)

            # for resnet_v2_50
            # heatmap data
            X0 = nets[0]['gap_conv']
            # heatmap weights
            W0 = gate.utils.analyzer.find_weights(
                'net1/resnet_v2_50/logits/weights')
            W0 = tf.reshape(W0, [-1])  # flatten

            for cur in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, mae, rmse, test_batch_info, X0, W0]
                _loss, _mae, _rmse, _info, x0, w0 = sess.run(feeds)

                mean_loss += _loss
                mean_mae += _mae
                mean_rmse += _rmse

                # generate heatmap
                x0 = np.transpose(x0, (0, 3, 1, 2))

                for _n in range(dataset.batch_size):

                    img_path = str(_info[_n], encoding='utf-8').split(' ')[0]
                    f_bn = os.path.basename(img_path)
                    p_name = re.findall('frames\\\(.*)\\\\', img_path)[0]
                    p_fold_path = os.path.join(heatmap_path, p_name)
                    if not os.path.exists(p_fold_path):
                        os.mkdir(p_fold_path)
                    fname = f_bn.split('.')[0] + '_' + global_step + '.jpg'

                    gate.utils.heatmap.single_map(
                        path=img_path, data=x0[_n], weight=w0,
                        raw_h=dataset.image.raw_height,
                        raw_w=dataset.image.raw_width,
                        save_path=os.path.join(p_fold_path, fname))

                logger.info('Has Processed %d of %d.' %
                            (cur * dataset.batch_size, dataset.total_num))

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter
            mean_rmse = 1.0 * mean_rmse / num_iter
            mean_mae = 1.0 * mean_mae / num_iter

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('mae', mean_mae, float)
            f_str.add('rmse', mean_rmse, float)
            logger.test(f_str.get())

            # for specify dataset
            # it use different compute method for mae/rmse
            # rewrite the mean_x value
            if dataset.name.find('avec2014') == 0:
                from project.avec2014 import avec2014_error
                mean_mae, mean_rmse = avec2014_error.get_accurate_from_file(
                    test_info_path, 'img')
                f_str = gate.utils.string.format_iter(global_step)
                f_str.add('loss', mean_loss, float)
                f_str.add('video_mae', mean_mae, float)
                f_str.add('video_rmse', mean_rmse, float)
                logger.test(f_str.get())

            return mean_mae, mean_rmse


def pipline(name, chkp_path, gen_heatmap=False):
    """ test all model in checkpoint file
    """
    import tools

    chkp_file_path = os.path.join(chkp_path, 'checkpoint')
    # make a backup
    gate.utils.filesystem.copy_file(chkp_file_path, chkp_file_path+'.bk')
    # acquire model list
    chkp_model_list = tools.checkpoint.get_checkpoint_model_items(chkp_file_path)
    # loop over the model list
    for idx in range(1, len(chkp_model_list)):
        # write list to new checkpoint file
        new_model_list = chkp_model_list.copy()
        new_model_list[0] = new_model_list[idx]
        tools.checkpoint.write_checkpoint_model_items(chkp_file_path, new_model_list)
        logger.info('Process model %s' % new_model_list[0])
        # run test_all
        if gen_heatmap:
            heatmap(name, chkp_path)
        else:
            test(name, chkp_path)