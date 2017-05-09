# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np


class WGAN():
    """ WGAN - updated: 2017/05/06
        Wasserstein GAN has four different point with DCGAN
        1) there is no sigmoid output of discriminator
        2) loss does not use log
        3) clip the gradients with constant c for discriminator
        4) use RMSProp or SGD optimizer
    """

    def __init__(self, dataset, is_training=True):
        # clip with a constant value
        self.clip_values_min = -0.01
        self.clip_values_max = 0.01

        # phase
        self.is_training = is_training

        # batch normalization param
        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-5

        # lr - rmsprop
        self.lr = dataset.lr.learning_rate

        # data related
        self.batch_size = dataset.batch_size
        self.sample_num = dataset.batch_size

        self.output_height = dataset.output_height
        self.output_width = dataset.output_width

        self.z_dim = 100
        self.y_dim = dataset.num_classes
        self.c_dim = dataset.channels

        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024

    # Component area
    def linear(self, x, output_dim):
        return layers.fully_connected(
            x, output_dim,
            activation_fn=None,
            biases_initializer=tf.constant_initializer(0.0),
            weights_initializer=tf.random_normal_initializer(
                stddev=0.02))

    def conv_cond_concat(self, x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def leak_relu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def batch_norm(self, x, is_training=True):
        return layers.batch_norm(
            x, decay=self.batch_norm_decay,
            updates_collections=None,
            is_training=is_training,
            epsilon=self.batch_norm_epsilon,
            scale=True)

    def conv2d(self, x, output_dim, name):
        return layers.conv2d(
            x, output_dim, [5, 5], 2, padding='SAME',
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
            biases_initializer=tf.constant_initializer(0.0),
            scope=name)

    def deconv2d(self, x, output_dim, name='deconv2d'):
        return layers.convolution2d_transpose(
            x, output_dim, [5, 5], 2, padding='SAME',
            activation_fn=None,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            biases_initializer=tf.constant_initializer(0.0))

    def model(self, global_step, inputs, labels, sample_z=None, sample_y=None):
        self.y = tf.to_float(tf.one_hot(labels, depth=self.y_dim, on_value=1))
        self.inputs = inputs
        self.z = tf.random_uniform(
            [self.batch_size, self.z_dim], minval=-1, maxval=1)

        # True data
        self.G = self.generator(self.z, self.y, False)
        self.D_logits = self.discriminator(inputs, self.y, False)

        # Fake data
        if sample_z is not None and sample_y is not None:
            self.sampler = self.generator(sample_z, sample_y, True, False)
        self.D_logits_ = self.discriminator(self.G, self.y, True)

        self.d_loss = tf.reduce_mean(self.D_logits - self.D_logits_)
        self.g_loss = tf.reduce_mean(self.D_logits_)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        # WGAN
        for var in self.d_vars:
            var = tf.clip_by_value(
                var, self.clip_values_min, self.clip_values_max)

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-')
        for var in self.d_vars:
            print(var)
        for var in self.g_vars:
            print(var)

        # pay attention
        # there global_step just running once
        d_optim = tf.train.RMSPropOptimizer(self.lr).minimize(
            global_step=global_step, loss=self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(self.lr).minimize(
            loss=self.g_loss, var_list=self.g_vars)

        return d_optim, g_optim

    def discriminator(self, image, y, reuse=False, name='discriminator'):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = self.conv_cond_concat(image, yb)

            h0_conv = self.conv2d(x, self.c_dim + self.y_dim, name='conv_h0')
            h0 = self.leak_relu(h0_conv)
            h0 = self.conv_cond_concat(h0, yb)

            h1_conv = self.conv2d(h0, self.df_dim + self.y_dim, name='conv_h1')
            h1 = self.leak_relu(self.batch_norm(h1_conv))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2_f = self.linear(h1, self.dfc_dim)
            h2 = self.leak_relu(self.batch_norm(h2_f))
            h2 = tf.concat([h2, y], 1)

            h3 = self.linear(h2, 1)

            return h3

    def generator(self, z, y, reuse=False, is_training=True, name='generator'):
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

            h0_f = self.linear(z, self.gfc_dim)
            h0 = tf.nn.relu(self.batch_norm(h0_f, is_training))
            h0 = tf.concat([h0, y], 1)

            h1_f = self.linear(h0, self.gf_dim * 2 * s_h4 * s_w4)
            h1 = tf.nn.relu(self.batch_norm(h1_f, is_training))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = self.conv_cond_concat(h1, yb)

            h2_f = self.deconv2d(h1, self.gf_dim * 2)
            h2 = tf.nn.relu(self.batch_norm(h2_f, is_training))
            h2 = self.conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(self.deconv2d(h2, self.c_dim))
