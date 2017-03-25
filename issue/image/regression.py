# -*- coding: utf-8 -*-
""" regression task for image
    updated: 2017/3/25
"""

import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

from gate import updater
from gate import utils
from gate import data
from gate import net

from gate.data import dataset_avec2014_utils


def train(data_name, net_name, chkp_path=None, exclusions=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = data.factory.get_dataset(data_name, 'train')

            # reset_training_path
            if chkp_path is not None:
                os.rmdir(dataset.log.train_dir)
                dataset.log.train_dir = chkp_path

            # ouput information
            utils.info.print_basic_information(dataset, net_name)

            # get data
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()

        logits, end_points = net.factory.get_network(
            net_name, 'train', images, 1)

        with tf.name_scope('error'):
            logits = tf.to_float(tf.reshape(logits, [dataset.batch_size, 1]))
            labels = tf.to_float(tf.reshape(labels, [dataset.batch_size, 1]))
            labels = tf.divide(labels, dataset.num_classes)
            losses = tf.nn.l2_loss([labels - logits], name='l2_loss')
            err_mae = tf.reduce_mean(
                input_tensor=tf.abs((logits - labels) * dataset.num_classes), name='err_mae')
            err_mse = tf.reduce_mean(
                input_tensor=tf.square((logits - labels) * dataset.num_classes), name='err_mse')

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        # optimizer
        with tf.device(dataset.device):
            learning_rate = updater.learning_rate.configure(dataset, dataset.total_num, global_step)
            optimizer = updater.optimizer.configure(dataset, learning_rate)

        # -------------------------------------------
        # Finetune Related
        #   if var appears in var_finetune, it will not be import.
        #      Commonly used for different number of output classes.
        # -------------------------------------------
        if exclusions is not None:
            variables_to_restore = []
            for var in tf.global_variables():
                excluded = False
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)
            saver = tf.train.Saver(var_list=variables_to_restore)
            variables_to_train = variables_to_restore
        else:
            saver = tf.train.Saver()
            variables_to_train = tf.trainable_variables()

        # compute gradients
        grads = tf.gradients(losses, variables_to_train)
        train_op = optimizer.apply_gradients(
            zip(grads, variables_to_train),
            global_step=global_step, name='train_step')

        # -------------------------------------------
        # Check point
        # -------------------------------------------
        chkp_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=dataset.log.train_dir,
            save_steps=dataset.log.save_model_iter,
            saver=tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10000),
            checkpoint_basename=dataset.name + '.ckpt')

        # -------------------------------------------
        # Summary Function
        # -------------------------------------------
        with tf.name_scope('train'):
            # iter must be the first scalar
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('lr', learning_rate)
            tf.summary.scalar('err_mae', err_mae)
            tf.summary.scalar('err_mse', err_mse)
            tf.summary.scalar('loss', losses)

        with tf.name_scope('grads'):
            for idx, v in enumerate(grads):
                prefix = variables_to_train[idx].name
                tf.summary.scalar(name=prefix + '_mean', tensor=tf.reduce_mean(v))
                tf.summary.scalar(name=prefix + '_max', tensor=tf.reduce_max(v))
                tf.summary.scalar(name=prefix + '_sum', tensor=tf.reduce_sum(v))

        summary_hook = tf.train.SummarySaverHook(
            save_steps=dataset.log.save_summaries_iter,
            output_dir=dataset.log.train_dir,
            summary_op=tf.summary.merge_all())

        summary_test = tf.summary.FileWriter(dataset.log.train_dir)

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0
                self.best_iter_mae, self.best_mae = 0.0, 1000.0
                self.best_iter_rmse, self.best_rmse = 0.0, 1000.0

            def begin(self):
                # continue to train
                print('[INFO] Loading in layer variable list as:')
                for v in variables_to_train:
                    print('[NET] ', v)

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
                    print(format_str % (cur_iter, _loss, _mae, _rmse, _lr, _duration))
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
                hooks=[chkp_hook, summary_hook, running_hook, tf.train.NanTensorHook(losses)],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            if chkp_path is not None:
                ckpt = tf.train.get_checkpoint_state(chkp_path)
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                print('[TRAIN] Load checkpoint from: %s' %
                      ckpt.model_checkpoint_path)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def test(name, net_name, model_path=None, summary_writer=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Preparing the dataset
        # -------------------------------------------
        dataset = data.factory.get_dataset(name, 'test')
        dataset.log.test_dir = model_path + '/test/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        utils.info.print_basic_information(dataset, net_name)

        images, labels_orig, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        logits, end_points = net.factory.get_network(
            net_name, 'test', images, 1)

        # ATTENTION!
        logits = tf.to_float(tf.reshape(logits, [dataset.batch_size, 1]))
        labels = tf.to_float(tf.reshape(labels_orig, [dataset.batch_size, 1]))
        labels = tf.div(labels, dataset.num_classes)
        losses = tf.nn.l2_loss([labels - logits], name='l2_loss')

        err_mae = tf.reduce_mean(input_tensor=tf.abs(
            (logits - labels) * dataset.num_classes), name='err_mae')
        err_mse = tf.reduce_mean(input_tensor=tf.square(
            (logits - labels) * dataset.num_classes), name='err_mse')

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
            logits_str = tf.as_string(tf.reshape(
                logits * dataset.num_classes, shape=[dataset.batch_size]))
            test_batch_info = filenames + tab + labels_str + tab + logits_str

            test_info_path = os.path.join(dataset.log.test_dir, '%s.txt' % global_step)

            test_info_fp = open(test_info_path, 'wb')
            print('[TEST] Output file in %s.' % test_info_path)

            # progressive bar
            progress_bar = utils.Progressive(min_scale=2.0)

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
            print('[TEST] Loss:%.4f, mae:%.4f, rmse:%.4f' % (loss, mae, rmse))

            # -------------------------------------------
            # Especially for avec2014
            # -------------------------------------------
            if name == 'avec2014' or name == 'avec2014_flow':
                mae, rmse = dataset_avec2014_utils.get_accurate_from_file(test_info_path)
                print('[TEST] Loss:%.4f, video_mae:%.4f, video_rmse:%.4f' % (loss, mae, rmse))

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
