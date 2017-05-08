# -*- coding: utf-8 -*-
""" CGAN - updated: 2017/05/05

For Mnist:
g_ -> generator
d_ -> discriminator
HyperParam
    epoch: 25
    learning_rate: 0.0002
    adam_beta1: 0.5
    train_size: np.inf
    batch_size: 64
    input_height: 28
    input_width: 28
    output_height: 28
    output_width: 28
    y_dim: 10
    c_dim: 1
    z_dim: 100
    gf_dim: 64
    df_dim: 64
    gfc_dim: 1024
    dfc_dim: 1024
    dataset: mnist
    input_fname_pattern: *.jpg
"""

import os
import time
import math
import re
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import framework

import gate

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

import numpy as np
from PIL import Image


class CGAN():

    def __init__(self, is_training=True):
        self.is_training = is_training

        self.lr = 0.0002
        self.adam_beta1 = 0.5

        # batch normalization param
        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-5

        self.batch_size = 64
        self.sample_num = 64

        self.input_height = 28
        self.input_width = 28
        self.output_height = 28
        self.output_width = 28

        self.z_dim = 100
        self.y_dim = 10
        self.c_dim = 3
        # self.image_dims = [28, 28, 1]

        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        self.sample_z = tf.convert_to_tensor(sample_z, dtype=tf.float32)

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

    def model(self, inputs, labels):
        self.y = labels
        self.inputs = inputs
        self.z = tf.random_uniform(
            [self.batch_size, self.z_dim], minval=-1, maxval=1)

        # True data
        self.G = self.generator(self.z, self.y, False)
        self.D, self.D_logits = self.discriminator(inputs, self.y, False)

        # Fake data
        self.sampler = self.generator(self.sample_z, self.y, True, False)
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

        print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#')
        for var in self.d_vars:
            print(var)
        for var in self.g_vars:
            print(var)

        self.saver = tf.train.Saver()

        d_optim = tf.train.AdamOptimizer(
            self.lr, beta1=self.adam_beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            self.lr, beta1=self.adam_beta1).minimize(self.g_loss, var_list=self.g_vars)

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

            return tf.nn.sigmoid(h3), h3

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


def train():
    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        dataset = gate.dataset.factory.get_dataset('cifar10', 'train')

        # get data
        images, labels, _ = dataset.loads()
        labels = tf.to_float(tf.one_hot(
            labels, depth=dataset.num_classes, on_value=1))

        # Net
        net = CGAN(is_training=True)
        d_optim, g_optim = net.model(images, labels)

        summary = tf.summary.FileWriter('test', graph=tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)

            for i in range(100000):
                # z, d_loss, _ = sess.run([net.z, net.d_loss, d_optim])
                # imgs, g_loss, _ = sess.run([net.G, net.g_loss, g_optim])
                # sess.run(g_optim)
                d_loss, _ = sess.run([net.d_loss, d_optim])
                g_loss, _ = sess.run([net.g_loss, g_optim])
                sess.run(g_optim)

                if i % 10 == 0:
                    imgs = sess.run(net.sampler)
                    save_images(
                        imgs, [8, 8], './{}/test_{:02d}_{:04d}.png'.format('test', 1, i))

                if i % 10 == 0:
                    print(i, d_loss, g_loss)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images + 1.) / 2.


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
