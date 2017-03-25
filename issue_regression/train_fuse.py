# -*- coding: utf-8 -*-
""" updated: 2017/3/23
"""

import os
import time
import math

import tensorflow as tf
import issue_regression.test_fuse as reg_test

from tensorflow.contrib import framework
from nets import nets_factory
from data import datasets_factory
from optimizer import opt_optimizer
from util_tools import output


def run(data_name, net_name, chkp_path=None, exclusions=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = datasets_factory.get_dataset(data_name, 'train')

            # reset_training_path
            if chkp_path is not None:
                os.rmdir(dataset.log.train_dir)
                dataset.log.train_dir = chkp_path

            # ouput information
            output.print_basic_information(dataset, net_name)

            # get data
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()

        logits1, end_points1 = nets_factory.get_network(
            'lightnet', 'train', images, 1)

        logits2, end_points2 = nets_factory.get_network(
            'alexnet', 'train', images, 1)

        with tf.name_scope('error'):

            logits1 = tf.to_float(tf.reshape(logits1, [dataset.batch_size, 1]))
            logits2 = tf.to_float(tf.reshape(logits2, [dataset.batch_size, 1]))

            # logits = tf.divide(tf.add(logits1, logits2), 2)
            logits = logits1
            labels = tf.to_float(tf.reshape(labels, [dataset.batch_size, 1]))
            labels = tf.divide(labels, dataset.num_classes)

            losses1 = tf.nn.l2_loss([labels - logits1], name='l2_loss_1')
            losses2 = tf.nn.l2_loss([labels - logits2], name='l2_loss_2')

            losses = losses1 + losses2

            err_mae = tf.reduce_mean(
                input_tensor=tf.abs((logits - labels) * dataset.num_classes), name='err_mae')
            err_mse = tf.reduce_mean(
                input_tensor=tf.square((logits - labels) * dataset.num_classes), name='err_mse')

        # -------------------------------------------
        # Gradients
        # -------------------------------------------
        # optimizer
        with tf.device(dataset.device):
            learning_rate = opt_optimizer.configure_learning_rate(
                dataset, dataset.total_num, global_step)
            optimizer = opt_optimizer.configure_optimizer(
                dataset, learning_rate)

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
            saver=tf.train.Saver(
                var_list=tf.global_variables(), max_to_keep=10000),
            checkpoint_basename=dataset.name + '.ckpt')

        # -------------------------------------------
        # Summary Function
        # -------------------------------------------
        with tf.name_scope('train'):
            tf.summary.scalar('lr', learning_rate)
            tf.summary.scalar('err_mae', err_mae)
            tf.summary.scalar('err_mse', err_mse)
            tf.summary.scalar('loss', losses)

        with tf.name_scope('grads'):
            for idx, v in enumerate(grads):
                prefix = variables_to_train[idx].name
                tf.summary.scalar(name=prefix + '_mean',
                                  tensor=tf.reduce_mean(v))
                tf.summary.scalar(name=prefix + '_max',
                                  tensor=tf.reduce_max(v))
                tf.summary.scalar(name=prefix + '_sum',
                                  tensor=tf.reduce_sum(v))

        summary_hook = tf.train.SummarySaverHook(
            save_steps=dataset.log.save_summaries_iter,
            output_dir=dataset.log.train_dir,
            summary_op=tf.summary.merge_all())

        summary_test = tf.summary.FileWriter(dataset.log.train_dir)

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class running_hook(tf.train.SessionRunHook):

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
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, mae:%.2f, rmse:%.2f, '
                    format_str += 'lr:%s, time:%.2fms.'
                    print(format_str %
                          (cur_iter, _loss, _mae, _rmse, _lr, _duration))
                    # set zero
                    self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_mae, test_rmse = reg_test.run(
                        data_name, net_name, dataset.log.train_dir, summary_test)
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
        running_hook = running_hook()

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook,
                       tf.train.NanTensorHook(losses)],
                save_summaries_steps=0,
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path) as mon_sess:

            if chkp_path is not None:
                ckpt = tf.train.get_checkpoint_state(chkp_path)
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                print('[TRAIN] Load checkpoint from: %s' %
                      ckpt.model_checkpoint_path)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
