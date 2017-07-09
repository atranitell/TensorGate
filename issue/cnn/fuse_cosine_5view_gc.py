# -*- coding: utf-8 -*-
""" fuse cosine task for image
    updated: 2017/06/20
"""
import os
import math
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework

import gate
from gate.utils.logger import logger
import gate.dataset.kinface.utils as kinface


def get_network(image1, image2, dataset, phase):

    img_F1, img_LE1, img_RE1, img_N1, img_M1 = image1
    img_F2, img_LE2, img_RE2, img_N2, img_M2 = image2

    _, end_pointsF1 = gate.net.factory.get_network(
        dataset.hps, phase, img_F1, 1, 'net_full1')
    _, end_pointsF2 = gate.net.factory.get_network(
        dataset.hps, phase, img_F2, 1, 'net_full2')

    _, end_pointsLE1 = gate.net.factory.get_network(
        dataset.hps, phase, img_LE1, 1, 'net_leye1')
    _, end_pointsLE2 = gate.net.factory.get_network(
        dataset.hps, phase, img_LE2, 1, 'net_leye2')

    _, end_pointsRE1 = gate.net.factory.get_network(
        dataset.hps, phase, img_RE1, 1, 'net_reye1')
    _, end_pointsRE2 = gate.net.factory.get_network(
        dataset.hps, phase, img_RE2, 1, 'net_reye2')

    _, end_pointsN1 = gate.net.factory.get_network(
        dataset.hps, phase, img_N1, 1, 'net_nose1')
    _, end_pointsN2 = gate.net.factory.get_network(
        dataset.hps, phase, img_N2, 1, 'net_nose2')

    _, end_pointsM1 = gate.net.factory.get_network(
        dataset.hps, phase, img_M1, 1, 'net_mouth1')
    _, end_pointsM2 = gate.net.factory.get_network(
        dataset.hps, phase, img_M2, 1, 'net_mouth2')

    feat_F1 = end_pointsF1['gap_pool']
    feat_F2 = end_pointsF2['gap_pool']

    feat_LE1 = end_pointsLE1['gap_pool']
    feat_LE2 = end_pointsLE2['gap_pool']

    feat_RE1 = end_pointsRE1['gap_pool']
    feat_RE2 = end_pointsRE2['gap_pool']

    feat_N1 = end_pointsN1['gap_pool']
    feat_N2 = end_pointsN2['gap_pool']

    feat_M1 = end_pointsM1['gap_pool']
    feat_M2 = end_pointsM2['gap_pool']

    nets = [end_pointsF1, end_pointsF2,
            end_pointsLE1, end_pointsLE2,
            end_pointsRE1, end_pointsRE2,
            end_pointsN1, end_pointsN2,
            end_pointsM1, end_pointsM2]

    feats = [feat_F1, feat_F2,
             feat_LE1, feat_LE2,
             feat_RE1, feat_RE2,
             feat_N1, feat_N2,
             feat_M1, feat_M2]

    return feats, nets


def get_loss(feats, img_gc, labels, batch_size, phase):
    """ get loss should receive target variables
            and output corresponding loss
    """
    is_training = True if phase is 'train' else False
    loss1, pred1 = gate.loss.cosine.get_loss(
        feats[0], feats[1], labels, batch_size, is_training)
    loss2, pred2 = gate.loss.cosine.get_loss(
        feats[2], feats[3], labels, batch_size, is_training)
    loss3, pred3 = gate.loss.cosine.get_loss(
        feats[4], feats[5], labels, batch_size, is_training)
    loss4, pred4 = gate.loss.cosine.get_loss(
        feats[6], feats[7], labels, batch_size, is_training)
    loss5, pred5 = gate.loss.cosine.get_loss(
        feats[8], feats[9], labels, batch_size, is_training)
    loss_gc, _ = gate.loss.cosine.get_loss(
        img_gc[0], img_gc[1], labels, batch_size, is_training)

    losses = loss1 + loss2 + loss3 + loss4 + loss5 + 0.01 * loss_gc
    preds = 0.2 * (pred1 + pred2 + pred3 + pred4 + pred5)

    loss = [loss1, loss2, loss3, loss4, loss5, loss_gc]
    pred = [pred1, pred2, pred3, pred4, pred5]

    return losses, preds, loss, pred


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
        img_F1, img_F2, img_LE1, img_LE2, \
            img_RE1, img_RE2, img_N1, img_N2, \
            img_M1, img_M2, img_gc1, img_gc2, \
            labels, fname1, fname2 = dataset.loads()

        image1 = img_F1, img_LE1, img_RE1, img_N1, img_M1
        image2 = img_F2, img_LE2, img_RE2, img_N2, img_M2
        img_gc = [img_gc1, img_gc2]

        # get global step
        global_step = framework.create_global_step()
        tf.summary.scalar('iter', global_step)

        # get network
        feats, _ = get_network(
            image1, image2, dataset, 'train')

        # get loss
        losses, _, _, _ = get_loss(
            feats, img_gc, labels, dataset.batch_size, 'train')

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
                    f_str.add('test_time', test_duration, float)
                    f_str.add('best_train_error', self.best_err, float)
                    f_str.add('test_error', self.best_err_t, float)
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
        img_F1, img_F2, img_LE1, img_LE2, \
            img_RE1, img_RE2, img_N1, img_N2, \
            img_M1, img_M2, img_gc1, img_gc2, \
            labels, fname1, fname2 = dataset.loads()

        image1 = img_F1, img_LE1, img_RE1, img_N1, img_M1
        image2 = img_F2, img_LE2, img_RE2, img_N2, img_M2
        img_gc = [img_gc1, img_gc2]

        # get network
        feats, _ = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions, _, _ = get_loss(
            feats, img_gc, labels, dataset.batch_size, 'test')

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
            mean_err = kinface.Error().get_result_from_file(
                test_info_path, threshold)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total_sample', dataset.total_num, int)
            f_str.add('num_batch', num_iter, int)
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
        img_F1, img_F2, img_LE1, img_LE2, \
            img_RE1, img_RE2, img_N1, img_N2, \
            img_M1, img_M2, img_gc1, img_gc2, \
            labels, fname1, fname2 = dataset.loads()

        image1 = img_F1, img_LE1, img_RE1, img_N1, img_M1
        image2 = img_F2, img_LE2, img_RE2, img_N2, img_M2
        img_gc = [img_gc1, img_gc2]

        # get network
        feats, _, _, _ = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions = get_loss(
            feats, img_gc, labels, dataset.batch_size, 'test')

        # get saver
        saver = tf.train.Saver(name='restore_all_test')

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
            mean_err, threshold = kinface.Error().get_result_from_file(test_info_path)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total_sample', dataset.total_num, int)
            f_str.add('num_batch', num_iter, int)
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
        img_F1, img_F2, img_LE1, img_LE2, \
            img_RE1, img_RE2, img_N1, img_N2, \
            img_M1, img_M2, img_gc1, img_gc2, \
            labels, fname1, fname2 = dataset.loads()

        image1 = img_F1, img_LE1, img_RE1, img_N1, img_M1
        image2 = img_F2, img_LE2, img_RE2, img_N2, img_M2
        img_gc = [img_gc1, img_gc2]

        # get network
        feats, nets = get_network(
            image1, image2, dataset, 'test')

        # get loss
        losses, predictions, loss_set, pred_set = get_loss(
            feats, img_gc, labels, dataset.batch_size, 'test')

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

            # full-view cosine similarity
            S = loss_set[0]
            X1 = nets[0]['gap_conv']
            X2 = nets[1]['gap_conv']
            X1_feat = nets[0]['gap_pool']
            X2_feat = nets[1]['gap_pool']

            # Start to TEST
            for cur in range(num_iter):
                if coord.should_stop():
                    break

                # running session to acuqire value
                feeds = [losses, test_batch_info, X1, X2, X1_feat, X2_feat, S]
                _loss, _info, x1, x2, x1_feat, x2_feat, score = sess.run(feeds)

                # generate heatmap
                # transpose dim for index
                x1 = np.transpose(x1, (0, 3, 1, 2))
                x2 = np.transpose(x2, (0, 3, 1, 2))


                # u/s , v/s
                # w1 = x2_feat/score
                # w2 = x1_feat/score

                for _n in range(dataset.batch_size):
                    _w1 = x1_feat[_n]
                    _w2 = x2_feat[_n]

                    norm_w1 = np.linalg.norm(_w1)
                    norm_w2 = np.linalg.norm(_w2)
                    norm_dot = norm_w1 * norm_w2

                    _w1 = _w2 / norm_dot
                    _w2 = _w1 / norm_dot

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
            mean_err, _ = kinface.Error().get_result_from_file(test_info_path)

            # output result
            f_str = gate.utils.string.format_iter(global_step)
            f_str.add('total_sample', dataset.total_num, int)
            f_str.add('num_batch', num_iter, int)
            f_str.add('loss', mean_loss, float)
            f_str.add('error', mean_err, float)
            logger.test(f_str.get())

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

            return mean_err
