# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import time

import tensorflow as tf
from tensorflow.contrib import framework

import issue_classification.test as class_test
from nets import nets_factory
from data import datasets_factory
from optimizer import opt_optimizer
from util_tools import output


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

        logits, end_points = nets_factory.get_network(
            net_name, 'train', images, dataset.num_classes)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        err = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        loss = tf.reduce_mean(losses)

        # add into summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc', err)

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
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, acc:%.2f, lr:%s, time:%.2fms'
                    print(format_str % (cur_iter, _loss, _err, _lr, _duration))
                    # set zero
                    self.loss, self.err, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    class_test.run(data_name, net_name, dataset.log.train_dir)

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
