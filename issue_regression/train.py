# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import time
import math

import tensorflow as tf
import issue_regression.test as reg_test

from tensorflow.contrib import framework
from nets import nets_factory
from data import datasets_factory
from optimizer import opt_optimizer
from util_tools import output


def run(data_name, net_name, chkp_path=None, var_trainable=None, var_finetune=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = datasets_factory.get_dataset(data_name, 'train')
            output.print_basic_information(dataset, net_name)

            # reset_training_path
            if chkp_path is not None:
                dataset.log.train_dir = chkp_path

            # get data
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()

        logits, end_points = nets_factory.get_network(
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

        # add into summary
        tf.summary.scalar('err_mae', err_mae)
        tf.summary.scalar('err_mse', err_mse)
        tf.summary.scalar('loss', losses)

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
        #   if var does not appeares in var_trainable, it will not be updated.
        #      Commonly used for fixed weights and bias for shallow layers.
        #   if var appears in var_finetune, it will not be import.
        #      Commonly used for different number of output classes.
        # -------------------------------------------
        if var_trainable is not None:
            variables_to_train = []
            for var_in_net in tf.trainable_variables():
                for var_in_list in var_trainable:
                    if var_in_net.name.startswith(var_in_list):
                        variables_to_train.append(var_in_net)
        else:
            variables_to_train = tf.trainable_variables()

        if var_finetune is not None:
            variables_to_finetune = tf.global_variables()
            for var_in_list in var_finetune:
                var_list = variables_to_finetune
                variables_to_finetune = []
                for var_in_net in var_list:
                    if not var_in_net.name.startswith(var_in_list):
                        variables_to_finetune.append(var_in_net)
            saver = tf.train.Saver(var_list=variables_to_finetune)

        # compute gradients
        grads = tf.gradients(losses, variables_to_train)
        train_op = optimizer.apply_gradients(
            zip(grads, variables_to_train),
            global_step=global_step, name='train_step')

        # add into summary
        tf.summary.scalar('lr', learning_rate)

        # -------------------------------------------
        # Check point
        # -------------------------------------------
        chkp_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=dataset.log.train_dir,
            save_steps=dataset.log.save_model_iter,
            saver=tf.train.Saver(var_list=tf.global_variables(),
                                 max_to_keep=10000),
            checkpoint_basename=dataset.name + '.ckpt')

        # -------------------------------------------
        # Summary Function
        # -------------------------------------------
        summary_hook = tf.train.SummarySaverHook(
            save_steps=dataset.log.save_summaries_iter,
            output_dir=dataset.log.train_dir,
            summary_op=tf.summary.merge_all()
        )

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class running_hook(tf.train.SessionRunHook):

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
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, mae:%.2f, rmse:%.2f, '
                    format_str += 'lr:%s, time:%.2fms.'
                    print(format_str % (cur_iter, _loss, _mae, _rmse, _lr, _duration))
                    # set zero
                    self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test_start_time = time.time()
                    test_mae, test_rmse = reg_test.run(data_name, net_name, dataset.log.train_dir)
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

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook()],
                save_summaries_steps=0,
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path) as mon_sess:

            # continue to train
            if chkp_path is not None:
                ckpt = tf.train.get_checkpoint_state(chkp_path)
                saver.restore(mon_sess, ckpt.model_checkpoint_path)
                print('[TRAIN] Load checkpoint from: %s' % ckpt.model_checkpoint_path)

            # output information
            if var_finetune is not None:
                print('[INFO] Loading in layer variable list as:')
                for v in variables_to_finetune:
                    print('[NET] ', v)

            print('[INFO] Trainable layer variable list as:')
            for v in variables_to_train:
                print('[NET] ', v)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
