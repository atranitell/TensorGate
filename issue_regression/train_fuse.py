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

# concat ops
# from tensorflow.python.ops import array_ops
# from tensorflow.contrib.framework import arg_scope
# from tensorflow.contrib import layers


def run(data_name, net_name, chkp_path=None):

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

        logits_cifarnet, cifarnet = nets_factory.get_network(
            'cifarnet', 'train', images, 1)

        logits_lenet, lenet = nets_factory.get_network(
            'lenet', 'train', images, 1)

        # net = array_ops.concat([cifarnet['logits'], lenet['Logits']], 1)

        # with arg_scope([layers.fully_connected],
        #                biases_initializer=tf.constant_initializer(0.1),
        #                weights_initializer=tf.truncated_normal_initializer(stddev=0.04),
        #                weights_regularizer=layers.l2_regularizer(0.0001),
        #                activation_fn=tf.nn.relu):
        #     logits = layers.fully_connected(net, 1, scope='last')
        with tf.name_scope('error'):
            labels = tf.to_float(tf.reshape(labels, [dataset.batch_size, 1]))
            labels = tf.divide(labels, dataset.num_classes)

            loss_cifarnet = tf.nn.l2_loss([labels - logits_cifarnet], name='l2_loss_cifarnet')
            loss_lenet = tf.nn.l2_loss([labels - logits_lenet], name='l2_loss_lenet')

            losses = loss_cifarnet + loss_lenet

            error = (logits_cifarnet + logits_lenet - 2.0 * labels) * dataset.num_classes / 2.0

            err_mae = tf.reduce_mean(input_tensor=tf.abs(error, name='err_mae'))
            err_mse = tf.reduce_mean(input_tensor=tf.square(error, name='err_mse'))

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

        # compute gradients
        variables_to_train = tf.trainable_variables()
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
                                 max_to_keep=10),
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
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, mae:%.2f, rmse:%.2f, lr:%s, time:%.2fms'
                    print(format_str % (cur_iter, _loss, _mae, _rmse, _lr, _duration))
                    # set zero
                    self.mean_loss, self.mean_mae, self.mean_mse, self.duration = 0, 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    reg_test.run(data_name, net_name, dataset.log.train_dir)

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook()],
                save_summaries_steps=0,
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
