# -*- coding: utf-8 -*-
""" Offer a set of setting for GAN's Family
Component:
    1. data provider
    2. GAN
        a. initilization
        b. Decoder, Generator, Discriminator, etc.
        c. Training
    3. record information
    4. save and load model
"""
import os
import time
import tensorflow as tf
from tensorflow.contrib import framework

import gate
import issue.gan.utils as utils


def train(data_name, net_name, chkp_path=None):

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

            # generate test data
            sample_z, sample_y = utils.generate_sample(
                dataset.batch_size, 8, dataset.num_classes, 100)

            # create gen folder
            imgfolder = os.path.join(dataset.log.train_dir, 'gen')
            if not os.path.exists(imgfolder):
                os.mkdir(imgfolder)

            # get data
            images, labels, _ = dataset.loads()

        # -------------------------------------------
        # GAN
        # -------------------------------------------
        with tf.device(dataset.device):
            global_step = framework.create_global_step()
            gan = gate.gan.factory.get_model(net_name, 'train', dataset)
            train_op = gan.model(global_step, images,
                                 labels, sample_z, sample_y)
            restore_saver = tf.train.Saver(
                var_list=tf.trainable_variables(),
                name='restore', allow_empty=True)

        # -------------------------------------------
        # Check point
        # -------------------------------------------
        with tf.name_scope('checkpoint'):
            snapshot = gate.solver.Snapshot()
            chkp_hook = snapshot.get_chkp_hook(dataset)
            summary_hook = snapshot.get_summary_hook(dataset)

        # -------------------------------------------
        # summary
        # -------------------------------------------
        with tf.name_scope('train'):
            # iter must be the first scalar
            tf.summary.scalar('iter', global_step)
            tf.summary.scalar('d_loss', gan.d_loss)
            tf.summary.scalar('g_loss', gan.g_loss)

        # -------------------------------------------
        # Running Info
        # -------------------------------------------
        class Running_Hook(tf.train.SessionRunHook):

            def __init__(self):
                self.mean_d_loss, self.mean_g_loss, self.duration = 0, 0, 0

            def after_create_session(self, session, coord):
                self.sess = session

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs(
                    [global_step, gan.d_loss, gan.g_loss],
                    feed_dict=None)

            def after_run(self, run_context, run_values):
                # accumulate datas
                cur_iter = run_values.results[0] - 1
                self.mean_d_loss += run_values.results[1]
                self.mean_g_loss += run_values.results[2]
                self.duration += (time.time() - self._start_time)

                # print information
                if cur_iter % dataset.log.print_frequency == 0:
                    _invl = dataset.log.print_frequency
                    _d_loss = self.mean_d_loss / _invl
                    _g_loss = self.mean_g_loss / _invl
                    _duration = self.duration * 1000 / _invl
                    # there time is the running time of a iteration
                    # (if 1 GPU, it is a batch)
                    format_str = 'Iter:%d, d_loss:%.4f, g_loss:%.4f, time:%.2fms.'
                    gate.utils.show.TRAIN(
                        format_str % (cur_iter, _d_loss, _g_loss, _duration))
                    # set zero
                    self.mean_d_loss, self.mean_g_loss, self.duration = 0, 0, 0

                # evaluation
                if cur_iter % dataset.log.test_interval == 0 and cur_iter != 0:
                    imgs = self.sess.run(gan.sampler)
                    img_path = os.path.join(
                        imgfolder, data_name + '_{:08d}.png'.format(cur_iter))
                    utils.save_images(imgs, [8, 8], img_path)

        # record running information
        running_hook = Running_Hook()

        # -------------------------------------------
        # Start to train
        # -------------------------------------------
        with tf.train.MonitoredTrainingSession(
                hooks=[chkp_hook, summary_hook, running_hook],
                config=tf.ConfigProto(allow_soft_placement=True),
                checkpoint_dir=chkp_path,
                save_checkpoint_secs=None,
                save_summaries_steps=None) as mon_sess:

            # load checkpoint
            if chkp_path is not None:
                snapshot.restore(mon_sess, chkp_path, restore_saver)

            while not mon_sess.should_stop():
                mon_sess.run(train_op)
