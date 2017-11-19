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
from tensorflow.contrib import layers

import gate
from gate.utils.logger import logger
import issue.gan.utils as utils


class CGAN():
    """ Conditional GAN
        g_ -> generator
        d_ -> discriminator
    """

    def __init__(self, dataset, is_training=True):
        # phase
        self.is_training = is_training

        # batch normalization param
        self.batch_norm_decay = 0.997
        self.batch_norm_epsilon = 1e-5

        # lr - adam
        self.lr = dataset.lr.learning_rate

        # data related
        self.batch_size = dataset.batch_size
        self.sample_num = dataset.batch_size

        self.output_height = dataset.image.output_height
        self.output_width = dataset.image.output_width

        self.z_dim = 100
        self.y_dim = dataset.num_classes
        self.c_dim = dataset.image.channels

        if self.output_height == 28:
            self.discriminator = self.discriminator_28x28
            self.generator = self.generator_28x28
        if self.output_height == 128:
            self.discriminator = self.discriminator_128x128
            self.generator = self.generator_128x128

        self.one_hot = dataset.one_hot

    # Component area
    def linear(self, x, output_dim, name='fc'):
        return layers.fully_connected(
            x, output_dim,
            activation_fn=None,
            biases_initializer=tf.constant_initializer(0.0),
            weights_initializer=tf.random_normal_initializer(
                stddev=0.02),
            scope=name)

    def conv_cond_concat(self, x, y, name='concat'):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3, name=name)

    def leak_relu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def batch_norm(self, x, is_training=True):
        return layers.batch_norm(
            x, decay=self.batch_norm_decay,
            updates_collections=None,
            is_training=is_training,
            epsilon=self.batch_norm_epsilon,
            scale=True)

    def conv2d(self, x, output_dim, name, ksize=5, stride=2):
        return layers.conv2d(
            x, output_dim, [ksize, ksize], stride, padding='SAME',
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=tf.constant_initializer(0.0),
            scope=name)

    def deconv2d(self, x, output_dim, name='deconv2d', ksize=5, stride=2):
        return layers.convolution2d_transpose(
            x, output_dim, [ksize, ksize], stride, padding='SAME',
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            biases_initializer=tf.constant_initializer(0.0),
            scope=name)

    def model(self, global_step, inputs, labels, sample_z=None, sample_y=None):
        """ inputs: (batchsize, h, w, c)
            labels: (batchsize, 1) [single label]
            sample_z: (batchsize, z_dim)
            sample_y: (batchsize, y_dim) [one_hot style]
        """
        if self.one_hot:
            self.y = tf.to_float(labels)
        else:
            self.y = tf.to_float(tf.one_hot(
                labels, depth=self.y_dim, on_value=1))

        print(self.y)
        self.inputs = inputs
        self.z = tf.random_uniform(
            [self.batch_size, self.z_dim], minval=-1, maxval=1)

        # True data
        self.G = self.generator(self.z, self.y, False)
        self.D, self.D_logits = self.discriminator(inputs, self.y, False)

        # Fake data
        if sample_z is not None and sample_y is not None:
            self.sampler = self.generator(sample_z, sample_y, True, False)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, True)

        # True data ->
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.D), logits=self.D_logits))
        # Fake data ->
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(self.D_), logits=self.D_logits_))

        # generator loss
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(self.D_), logits=self.D_logits_))
        # discriminator loss
        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        for var in self.d_vars:
            print(var)
        for var in self.g_vars:
            print(var)

        # pay attention
        # there global_step just running once
        d_optim = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss=self.d_loss, global_step=global_step, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss=self.g_loss, var_list=self.g_vars)

        return [d_optim, g_optim]

    def discriminator_28x28(self, image, y, reuse=False, name='discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = self.conv_cond_concat(image, yb)

            h0_conv = self.conv2d(x, self.c_dim + self.y_dim, name='conv_h0')
            h0 = self.leak_relu(h0_conv)
            h0 = self.conv_cond_concat(h0, yb)

            h1_conv = self.conv2d(
                h0, self.batch_size + self.y_dim, name='conv_h1')
            h1 = self.leak_relu(self.batch_norm(h1_conv))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2_f = self.linear(h1, 1024, 'fc2')
            h2 = self.leak_relu(self.batch_norm(h2_f))
            h2 = tf.concat([h2, y], 1)

            h3 = self.linear(h2, 1, 'fc3')

            return tf.nn.sigmoid(h3), h3

    def generator_28x28(self, z, y, reuse=False, is_training=True, name='generator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            # 28, 28
            s_h, s_w = self.output_height, self.output_width
            # 14, 7
            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
            # 14, 7
            s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            h0_f = self.linear(z, 1024, 'fc0')
            h0 = tf.nn.relu(self.batch_norm(h0_f, is_training))
            h0 = tf.concat([h0, y], 1)

            h1_f = self.linear(h0, self.batch_size * 2 * s_h4 * s_w4, 'fc1')
            h1 = tf.nn.relu(self.batch_norm(h1_f, is_training))
            h1 = tf.reshape(h1, [self.batch_size, s_h4,
                                 s_w4, self.batch_size * 2])

            h1 = self.conv_cond_concat(h1, yb)

            h2_f = self.deconv2d(h1, self.batch_size * 2, 'fc2')
            h2 = tf.nn.relu(self.batch_norm(h2_f, is_training))
            h2 = self.conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(self.deconv2d(h2, self.c_dim))

    def discriminator_128x128(self, image, y, reuse=False, name='discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('d_block_init'):
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = self.conv_cond_concat(image, yb, name='d_concat_x')

            # 128x128
            with tf.variable_scope('d_block0'):
                h0_conv = self.conv2d(
                    x, self.c_dim + self.y_dim, name='d_conv0')
                h0 = self.leak_relu(h0_conv)
                h0 = self.conv_cond_concat(h0, yb, 'd_cond_concat_0')

            # 64x64
            with tf.variable_scope('d_block1'):
                h1_conv = self.conv2d(h0, 64 + self.y_dim, name='d_conv1')
                h1 = self.leak_relu(self.batch_norm(h1_conv))
                h1 = self.conv_cond_concat(h1, yb, 'd_cond_concat_1')

            # 32x32
            with tf.variable_scope('d_block2'):
                h2_conv = self.conv2d(h1, 92 + self.y_dim, name='d_conv2')
                h2 = self.leak_relu(self.batch_norm(h2_conv))
                h2 = self.conv_cond_concat(h2, yb, 'd_cond_concat_2')

            # 16x16
            with tf.variable_scope('d_block3'):
                h3_conv = self.conv2d(h2, 128 + self.y_dim, name='d_conv3')
                h3 = self.leak_relu(self.batch_norm(h3_conv))
                h3 = self.conv_cond_concat(h3, yb, 'd_cond_concat_3')

            # 8x8
            with tf.variable_scope('d_block4'):
                h4_conv = self.conv2d(h3, 256 + self.y_dim, name='d_conv4')
                h4 = self.leak_relu(self.batch_norm(h4_conv))
                h4 = self.conv_cond_concat(h4, yb, 'd_cond_concat_4')

            # 4x4
            with tf.variable_scope('d_block5'):
                h5_conv = self.conv2d(h4, 256 + self.y_dim, name='d_conv5')
                h5 = self.leak_relu(self.batch_norm(h5_conv))
                h5 = tf.reshape(h5, [self.batch_size, -1])

            # linear1
            with tf.variable_scope('d_block6'):
                h6_f = self.linear(h5, 1024, name='d_fc_6')
                h6 = self.leak_relu(self.batch_norm(h6_f))
                h6 = tf.concat([h6, y], 1, name='d_concat_6')

            # linear2
            with tf.variable_scope('d_block7'):
                h7 = self.linear(h6, 1, name='d_fc_7')
                return tf.nn.sigmoid(h7), h7

    def generator_128x128(self, z, y, reuse=False, is_training=True, name='generator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            with tf.variable_scope('g_block_init'):
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat([z, y], 1, name='g_concat_z')

            with tf.variable_scope('g_block0'):
                h0_f = self.linear(z, 1024, name='g_h0_f')
                h0 = tf.nn.relu(self.batch_norm(
                    h0_f, is_training), name='g_relu_h0')
                h0 = tf.concat([h0, y], 1, name='g_concat_0')

            with tf.variable_scope('g_block1'):
                h1_f = self.linear(h0, 1024, name='g_h1_f')
                h1 = tf.nn.relu(self.batch_norm(
                    h1_f, is_training), name='g_relu_h1')
                h1 = tf.reshape(h1, [self.batch_size, 2, 2, 256])
                h1 = self.conv_cond_concat(h1, yb, 'g_cond_concat_1')

            # 2x2
            with tf.variable_scope('g_block2'):
                c1_f = self.deconv2d(h1, 256, ksize=3, name='g_deconv_1')
                c1 = tf.nn.relu(self.batch_norm(
                    c1_f, is_training), name='g_relu_c1')
                c1 = self.conv_cond_concat(c1, yb, 'g_cond_concat_2')

            # 4x4
            with tf.variable_scope('g_block3'):
                c2_f = self.deconv2d(c1, 256, ksize=3, name='g_deconv_2')
                c2 = tf.nn.relu(self.batch_norm(
                    c2_f, is_training), name='g_relu_c2')
                c2 = self.conv_cond_concat(c2, yb, 'g_cond_concat_3')

            # 8x8
            with tf.variable_scope('g_block4'):
                c3_f = self.deconv2d(c2, 128, ksize=5, name='g_deconv_3')
                c3 = tf.nn.relu(self.batch_norm(
                    c3_f, is_training), name='g_relu_c3')
                c3 = self.conv_cond_concat(c3, yb, 'g_cond_concat_4')

            # 16x16
            with tf.variable_scope('g_block5'):
                c4_f = self.deconv2d(c3, 92, ksize=5, name='g_deconv_4')
                c4 = tf.nn.relu(self.batch_norm(
                    c4_f, is_training), name='g_relu_c4')
                c4 = self.conv_cond_concat(c4, yb, 'g_cond_concat_5')

            # 32x32
            with tf.variable_scope('g_block6'):
                c5_f = self.deconv2d(c4, 64, ksize=5, name='g_deconv_5')
                c5 = tf.nn.relu(self.batch_norm(
                    c5_f, is_training), name='g_relu_c5')
                c5 = self.conv_cond_concat(c5, yb, 'g_cond_concat_6')

            # 64x64
            return tf.nn.sigmoid(self.deconv2d(c5, self.c_dim, name='g_deconv_5'))


def train(data_name, chkp_path=None):

    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        with tf.name_scope('dataset'):
            dataset = gate.dataset.factory.get_dataset(
                data_name, 'train', chkp_path)

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
        global_step = framework.create_global_step()
        gan = CGAN(dataset, 'train')
        train_op = gan.model(global_step, images, labels, sample_z, sample_y)
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
                    logger.train(
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


def test(data_name, chkp_path):
    """ output generated image
        if sample_dict is None, it will generate sample by itself.
    """
    with tf.Graph().as_default():
        # Initail Data related
        dataset = gate.dataset.factory.get_dataset(
            data_name, 'test', chkp_path)

        # generate test data
        sample_z, sample_y = utils.generate_sample(
            dataset.batch_size, 8, dataset.num_classes, 100)

        utils.generate_all_class(dataset.num_classes, dataset.num_classes, 100)
        print(sample_z, sample_y)

        # GAN
        global_step = framework.create_global_step()
        gan = CGAN(dataset, 'test')
        sampler = gan.generator(sample_z, sample_y, False, False)

        # start
        saver = tf.train.Saver(name='restore_all_test')
        with tf.Session() as sess:

            utils.save_sampler(sess, sample_z, sample_y)

            # restore from checkpoint
            snapshot = gate.solver.Snapshot()
            global_step = snapshot.restore(sess, chkp_path, saver)

            imgs = sess.run(sampler)
            img_path = os.path.join(
                dataset.log.test_dir,
                data_name + '_{:08d}.png'.format(int(global_step)))
            utils.save_images(imgs, [8, 8], img_path)
