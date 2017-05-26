# -*- coding: utf-8 -*-
""" fuse cosine task for image
    updated: 2017/05/19

    X->[Inception_Resnet_V1(face)-freeze]->X1(N, 1792)->[mlp-train]->X2(N, 512)
    Y->[Inception_Resnet_V1(face)-freeze]->Y1(N, 1792)->[mlp-train]->Y2(N, 512)
    [X2, Y2](N, 512*2) -> [cosine loss]

"""
import os
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework

import gate
from gate.utils.logger import logger
import project.kinface.kinface_distance as kinface

from tensorflow.contrib import slim


def get_network(image1, image2, dataset, phase):
    # use Inception-Resnet-V1
    def _avg_pool(net, size, scope):
        with tf.variable_scope(scope):
            slim.avg_pool2d
            net = slim.avg_pool2d(
                net, size, 1, padding='VALID', scope=scope)
            return slim.flatten(net)

    def _build_net(image, dataset, phase, scope, reuse):
        dataset.hps.net_name = 'inception_resnet_v1'
        _, end_points1 = gate.net.factory.get_network(
            dataset.hps, 'test', image, 128, '', reuse=reuse)
        # merge multi-way
        data_1 = _avg_pool(
            end_points1['Conv2d_4b_3x3'], 17, scope + '/AvgPool_17x17')
        data_2 = _avg_pool(
            end_points1['Mixed_6a'], 8, scope + '/AvgPool_8x8')
        # data_3 = _avg_pool(
        #     end_points1['Mixed_7a'], 3, scope + '/AvgPool_3x3')
        data_3 = end_points1['PostPool']

        # data = tf.concat([data_1, data_2, data_3], axis=1)

        dataset.hps.net_name = 'mlp'
        logits1, end_points2 = gate.net.factory.get_network(
            dataset.hps, phase, data_1, dataset.num_classes, scope + '1')
        logits2, end_points2 = gate.net.factory.get_network(
            dataset.hps, phase, data_2, dataset.num_classes, scope + '2')
        logits3, end_points2 = gate.net.factory.get_network(
            dataset.hps, phase, data_3, dataset.num_classes, scope + '3')

        logits = [logits1, logits2, logits3]
        end_points = [end_points1, end_points2]

        return logits, end_points2

    logits1, net1 = _build_net(image1, dataset, phase, 'net1', False)
    logits2, net2 = _build_net(image2, dataset, phase, 'net2', True)

    return logits1, logits2, net1, net2


def get_loss(logits1, logits2, labels, batch_size, phase):
    """ get loss should receive target variables
            and output corresponding loss
    """
    is_training = True if phase is 'train' else False
    losses_0, predictions_0 = gate.loss.cosine.get_loss(
        logits1[0], logits2[0], labels, batch_size, is_training)
    losses_1, predictions_1 = gate.loss.cosine.get_loss(
        logits1[1], logits2[1], labels, batch_size, is_training)
    losses_2, predictions_2 = gate.loss.cosine.get_loss(
        logits1[2], logits2[2], labels, batch_size, is_training)

    losses = losses_0 + losses_1 + losses_2
    predictions = (predictions_0 + predictions_1 + predictions_2) * 0.3333333
    return losses, predictions


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
        image1, image2, labels, fname1, fname2 = dataset.loads()

        # get global step
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # get network
        logits1, logits2, net1, net2 = get_network(
            image1, image2, dataset, 'train')

        # get loss
        losses, predictions = get_loss(
            logits1, logits2, labels, dataset.batch_size, 'train')

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
                self.best_iter, self.best_err, self.best_err_t = 0, 0, 0

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

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('loss', _loss, float)
                    f_str.add('lr', _lr)
                    f_str.add('time', _duration, float)
                    logger.train(f_str.get())

                    # set zero
                    self.mean_loss, self.duration = 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    # acquire best threshold
                    trn_err, threshold = val(data_name, dataset.log.train_dir,
                                             summary_test)
                    # acquire test error
                    test_start_time = time.time()
                    test_err = test(data_name, dataset.log.train_dir,
                                    threshold, summary_test)
                    test_duration = time.time() - test_start_time
                    # according to trn err to pinpoint test err
                    if trn_err > self.best_err:
                        self.best_err = trn_err
                        self.best_err_t = test_err
                        self.best_iter = cur_iter

                    f_str = gate.utils.string.format_iter(cur_iter)
                    f_str.add('test time', test_duration, float)
                    f_str.add('best train error', self.best_err, float)
                    f_str.add('test error', self.best_err_t, float)
                    f_str.add('in', self.best_iter, int)
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


def test(data_name, chkp_path, threshold, summary_writer=None):
    """ test
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'test', chkp_path)

        # build data model
        image1, image2, labels, fname1, fname2 = dataset.loads()

        # get network
        logits1, logits2, nets1, nets2 = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions = get_loss(
            logits1, logits2, labels, dataset.batch_size, 'test')

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
            mean_loss = 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            preds_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            test_batch_info = fname1 + tab + fname2 + tab
            test_batch_info += labels_str + tab + preds_str

            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for cur in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, test_batch_info]
                _loss, _info = sess.run(feeds)
                mean_loss += _loss

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter

            # acquire actual accuracy
            mean_err = kinface.get_kinface_error(test_info_path, threshold)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('error', mean_err, float)
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


def val(data_name, chkp_path, summary_writer=None):
    """ acquire best cosine value
    """
    with tf.Graph().as_default():
        # get dataset
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'val', chkp_path)

        # build data model
        image1, image2, labels, fname1, fname2 = dataset.loads()

        # get network
        logits1, logits2, nets1, nets2 = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions = get_loss(
            logits1, logits2, labels, dataset.batch_size, 'test')

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
            mean_loss = 0

            # output to file
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            preds_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            test_batch_info = fname1 + tab + fname2 + tab
            test_batch_info += labels_str + tab + preds_str

            test_info_path = os.path.join(
                dataset.log.val_dir, '%s.txt' % global_step)
            test_info_fp = open(test_info_path, 'wb')

            for cur in range(num_iter):
                # if ctrl-c
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, test_batch_info]
                _loss, _info = sess.run(feeds)
                mean_loss += _loss

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            # stop
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
            test_info_fp.close()

            # statistic
            mean_loss = 1.0 * mean_loss / num_iter

            # acquire actual accuracy
            mean_err, threshold = kinface.get_kinface_error(test_info_path)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('error', mean_err, float)
            logger.val(f_str.get())

            # write to summary
            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='val/iter', simple_value=int(global_step))
                summary.value.add(tag='val/err', simple_value=mean_err)
                summary.value.add(tag='val/loss', simple_value=mean_loss)
                summary_writer.add_summary(summary, global_step)

            return mean_err, threshold


def heatmap(name, chkp_path):
    """ generate heatmap
    """
    with tf.Graph().as_default():
        # Preparing the dataset
        dataset = gate.dataset.factory.get_dataset(name, 'test', chkp_path)
        dataset.log.test_dir = chkp_path + '/test_heatmap/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        # build data model
        image1, image2, labels, fname1, fname2 = dataset.loads()

        # get network
        logits1, logits2, nets1, nets2 = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions = get_loss(
            logits1, logits2, labels, dataset.batch_size, 'test')

        # restore from checkpoint
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
            mean_loss = 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(
                labels, shape=[dataset.batch_size]))
            preds_str = tf.as_string(tf.reshape(
                predictions, shape=[dataset.batch_size]))
            test_batch_info = fname1 + tab + fname2 + tab
            test_batch_info += labels_str + tab + preds_str

            # open log file
            test_info_path = os.path.join(
                dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')

            # -------------------------------------------------
            # heatmap related
            heatmap_path = chkp_path + '/heatmap/'
            if not os.path.exists(heatmap_path):
                os.mkdir(heatmap_path)

            # heatmap weights
            W1 = gate.utils.analyzer.find_weights('net1/MLP/fc1/weights')
            W2 = gate.utils.analyzer.find_weights('net2/MLP/fc1/weights')

            # heatmap data
            X1 = nets1[0]['PrePool']
            X2 = nets2[0]['PrePool']

            # Start to TEST
            for cur in range(num_iter):
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, test_batch_info,
                         X1, W1, X2, W2, logits1, logits2]
                _loss, _info, x1, w1, x2, w2, l1, l2 = sess.run(feeds)

                # generate heatmap
                # transpose dim for index
                x1 = np.transpose(x1, (0, 3, 1, 2))
                x2 = np.transpose(x2, (0, 3, 1, 2))
                w1 = np.transpose(w1, (1, 0))
                w2 = np.transpose(w2, (1, 0))

                for _n in range(dataset.batch_size):

                    _, pos = gate.utils.math.find_max(l1[_n], l2[_n])
                    _w1 = w1[pos]
                    _w2 = w2[pos]

                    img1 = str(_info[_n], encoding='utf-8').split(' ')[0]
                    img2 = str(_info[_n], encoding='utf-8').split(' ')[1]

                    f1_bn = os.path.basename(img1)
                    f2_bn = os.path.basename(img2)
                    fname1 = f1_bn.split('.')[0] + '_' + global_step + '.jpg'
                    fname2 = f2_bn.split('.')[0] + '_' + global_step + '.jpg'

                    gate.utils.heatmap.single_map(
                        path=img1, data=x1[_n], weight=_w1,
                        raw_h=dataset.image.raw_height,
                        raw_w=dataset.image.raw_width,
                        save_path=os.path.join(heatmap_path, fname1))

                    gate.utils.heatmap.single_map(
                        path=img2, data=x2[_n], weight=_w2,
                        raw_h=dataset.image.raw_height,
                        raw_w=dataset.image.raw_width,
                        save_path=os.path.join(heatmap_path, fname2))

                logger.info('Has Processed %d of %d.' %
                            (cur * dataset.batch_size, dataset.total_num))

                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')

            test_info_fp.close()
            mean_loss = 1.0 * mean_loss / num_iter

            # output - acquire actual accuracy
            mean_err, _ = kinface.get_kinface_error(test_info_path)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total sample', dataset.total_num, int)
            f_str.add('num batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('error', mean_err, float)
            logger.test(f_str.get())

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return mean_err
