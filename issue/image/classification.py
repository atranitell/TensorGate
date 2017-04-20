# -*- coding: utf-8 -*-
""" updated: 2017/3/30
"""
import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

import gate


def train(data_name, net_name, chkp_path=None, exclusions=None):
    """ train for classification
    """

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
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()
            logits, _ = gate.net.factory.get_network(
                net_name, 'train', images, dataset.num_classes)

        with tf.name_scope('loss'):
            logits = tf.to_float(tf.reshape(
                logits, [dataset.batch_size, dataset.num_classes]))
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            loss = tf.reduce_mean(losses)

        with tf.name_scope('error'):
            predictions = tf.to_int32(tf.argmax(logits, axis=1))
            err = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))

        with tf.name_scope('train'):
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('err', err)
            tf.summary.scalar('loss', loss)

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        with tf.device(dataset.device):
            updater = gate.solver.Updater()
            updater.init_default_updater(
                dataset, global_step, loss, exclusions)

            learning_rate = updater.get_learning_rate()
            saver = updater.get_variables_saver()
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
        class running_hook(tf.train.SessionRunHook):
            """ running hook information
            """

            def __init__(self):
                self.loss, self.err, self.duration = 0, 0, 0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, err, loss, learning_rate],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.err += run_values.results[1]
                self.loss += run_values.results[2]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _err = self.err / _invl
                    _loss = self.loss / _invl
                    _lr = str(run_values.results[3])
                    _duration = self.duration * 1000 / _invl
                    # there time is the running time of a iteration
                    # (if 1 GPU, it is a batch)
                    format_str = (
                        'Iter:%d, loss:%.4f, acc:%.4f, lr:%s, time:%.2fms' %
                        (cur_iter, _loss, _err, _lr, _duration))
                    gate.utils.show.TRAIN(format_str)
                    # set zero
                    self.loss, self.err, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test(data_name, net_name, dataset.log.train_dir,
                         summary_test)

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook()],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def test(name, net_name, chkp_path=None, summary_writer=None):
    """ test for classification
    """

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(name, 'test')
            dataset.log.test_dir = chkp_path + '/test/'
            if not os.path.exists(dataset.log.test_dir):
                os.mkdir(dataset.log.test_dir)

            images, labels_orig, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            logits, _ = gate.net.factory.get_network(
                net_name, 'test', images, dataset.num_classes)

        with tf.name_scope('loss'):
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

            # progressive bar
            progress_bar = gate.utils.Progressive(min_scale=2.0)

            # Start to TEST
            for cur in range(num_iter):
                if coord.should_stop():
                    break
                # running session to acuqire value
                _loss, _err, _info = sess.run([loss, err, test_batch_info])
                output_loss += _loss
                output_err += _err
                # save tensor info to text file
                for _line in _info:
                    test_info_fp.write(_line + b'\r\n')
                # show the progressive bar, in percentage
                progress_bar.add_float(cur, num_iter)

            test_info_fp.close()

            output_loss = 1.0 * output_loss / num_iter
            output_err = 1.0 * output_err / num_iter

            # -------------------------------------------
            # output
            # -------------------------------------------
            print()
            format_str = 'Iter:%d, total test sample:%d, num_batch:%d' % \
                (int(global_step), dataset.total_num, num_iter)
            gate.utils.show.TEST(format_str)
            gate.utils.show.TEST('Loss:%.4f, acc:%.4f' %
                                 (output_loss, output_err))

            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(
                    tag='test/iter', simple_value=int(global_step))
                summary.value.add(tag='test/acc', simple_value=output_err)
                summary.value.add(tag='test/loss', simple_value=output_loss)
                summary_writer.add_summary(summary, global_step)

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
