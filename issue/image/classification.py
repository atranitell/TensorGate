# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""
import os
import time
import math

import tensorflow as tf
from tensorflow.contrib import framework

from gate import updater
from gate import utils
from gate import datains
from gate import net


def train(data_name, net_name, chkp_path=None, exclusions=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = datains.factory.get_dataset(data_name, 'train')

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
            net_name, 'train', images, dataset.num_classes)

        logits = tf.to_float(tf.reshape(logits, [dataset.batch_size, dataset.num_classes]))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        err = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
        loss = tf.reduce_mean(losses)

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
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('lr', learning_rate)
            tf.summary.scalar('err', err)
            tf.summary.scalar('loss', loss)

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
        class running_hook(tf.train.SessionRunHook):

            def __init__(self):
                self.loss, self.err, self.duration = 0, 0, 0

            def begin(self):
                # continue to train
                print('[INFO] Loading in layer variable list as:')
                for v in variables_to_train:
                    print('[NET] ', v)

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
                    format_str = '[TRAIN] Iter:%d, loss:%.4f, acc:%.4f, lr:%s, time:%.2fms'
                    print(format_str % (cur_iter, _loss, _err, _lr, _duration))
                    # set zero
                    self.loss, self.err, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    test(data_name, net_name, dataset.log.train_dir, summary_test)

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
        dataset = datains.factory.get_dataset(name, 'test')
        dataset.log.test_dir = model_path + '/test/'
        if not os.path.exists(dataset.log.test_dir):
            os.mkdir(dataset.log.test_dir)

        utils.info.print_basic_information(dataset, net_name)

        images, labels_orig, filenames = dataset.loads()

        # -------------------------------------------
        # Network
        # -------------------------------------------
        logits, end_points = net.factory.get_network(
            net_name, 'test', images, dataset.num_classes)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_orig, logits=logits)

        predictions = tf.to_int32(tf.argmax(logits, axis=1))
        err = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels_orig)))
        loss = tf.reduce_mean(losses)

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
            output_err, output_loss = 0, 0

            # output test information
            tab = tf.constant(' ', shape=[dataset.batch_size])
            labels_str = tf.as_string(tf.reshape(labels_orig, shape=[dataset.batch_size]))
            logits_str = tf.as_string(tf.reshape(predictions, shape=[dataset.batch_size]))
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
            print('\n[TEST] Iter:%d, total test sample:%d, num_batch:%d' %
                  (int(global_step), dataset.total_num, num_iter))

            print('[TEST] Loss:%.4f, acc:%.4f' % (output_loss, output_err))

            summary = tf.Summary()
            summary.value.add(tag='test/iter', simple_value=int(global_step))
            summary.value.add(tag='test/acc', simple_value=output_err)
            summary.value.add(tag='test/loss', simple_value=output_loss)
            summary_writer.add_summary(summary, global_step)

            # -------------------------------------------
            # terminate all threads
            # -------------------------------------------
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
