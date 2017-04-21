# -*- coding: utf-8 -*-
""" dcgan
    updated: 2017/04/17
"""

import os
import time
import math
import re

import tensorflow as tf
from tensorflow.contrib import framework

import gate

from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers

import numpy as np
from PIL import Image


class dcgan():

    def __init__(self):
        self.weight_decay = 0.0001
        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-5
        self.batch_norm_scale = True

    def arg_scope(self):
        weight_decay = self.weight_decay
        batch_norm_decay = self.batch_norm_decay
        batch_norm_epsilon = self.batch_norm_epsilon
        batch_norm_scale = self.batch_norm_scale

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': None,
            'zero_debias_moving_mean': True
        }

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       weights_regularizer=None,
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=0.02),
                       biases_initializer=tf.constant_initializer(0.0),
                       activation_fn=None,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       padding='SAME'):
            with arg_scope([layers.batch_norm], **batch_norm_params):
                with arg_scope([layers.max_pool2d, layers.avg_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    def leak_relu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def discriminator(self, images, num_classes, is_training):
        with tf.variable_scope('d_'):
            with arg_scope(self.arg_scope()):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    # for image 32 * 32 * 1
                    net = self.leak_relu(layers.conv2d(images, 64, [5, 5], 2))
                    # 16 * 16 * 64
                    net = self.leak_relu(layers.conv2d(net, 128, [5, 5], 2))
                    # 8 * 8 * 128
                    net = self.leak_relu(layers.conv2d(net, 256, [5, 5], 2))
                    # 4 * 4 * 256
                    net = self.leak_relu(layers.flatten(net))
                    # 16
                    logits = layers.fully_connected(
                        net, num_classes,
                        biases_initializer=tf.zeros_initializer(),
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.02),
                        weights_regularizer=None,
                        activation_fn=tf.nn.sigmoid,
                        scope='logits')
                    return logits

    def generator(self, input, is_training):
        with tf.variable_scope('g_'):
            with arg_scope(self.arg_scope()):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    net = layers.fully_connected(
                        input, 4 * 4 * 128,
                        biases_initializer=tf.zeros_initializer(),
                        weights_initializer=tf.truncated_normal_initializer(
                            stddev=0.02),
                        weights_regularizer=None,
                        activation_fn=None,
                        scope='de-fc')
                    net = tf.reshape(net, [-1, 4, 4, 128])
                    net = tf.nn.relu(net)
                    net = layers.conv2d_transpose(
                        net, 64, [5, 5], 2, padding='SAME')
                    net = tf.nn.relu(net)
                    net = layers.conv2d_transpose(
                        net, 32, [5, 5], 2, padding='SAME')
                    net = tf.nn.relu(net)
                    net = layers.conv2d_transpose(
                        net, 3, [5, 5], 2, padding='SAME')
                    return tf.nn.sigmoid(net)


def train():
    with tf.Graph().as_default():
        # -------------------------------------------
        # Initail Data related
        # -------------------------------------------
        dataset = gate.dataset.factory.get_dataset('mnist', 'train')

        # get data
        images, labels, _ = dataset.loads()

        global_step = framework.create_global_step()

        DCGAN_NET = dcgan()
        fake_input = tf.random_uniform([32, 100])
        # fake_input = tf.placeholder(tf.float32, [32, 100])
        G_net = DCGAN_NET.generator(fake_input, True)
        D_real = DCGAN_NET.discriminator(images, 2, True)
        D_fake = DCGAN_NET.discriminator(G_net, 2, True)

        # loss
        D_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real, labels=tf.ones_like(D_real)))
        D_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake, labels=tf.zeros_like(D_fake)))

        d_loss = D_real_loss + D_fake_loss
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake, labels=tf.ones_like(D_fake)))

        for weight in tf.trainable_variables():
            gate.utils.show.NET(str(weight))

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        d_optim = tf.train.AdamOptimizer(
            0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(
            0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.start_queue_runners(sess=sess)
            # print(sess.run(D_real))
            # print(sess.run(D_fake))cls
            for i in range(100000):
                if i % 1000 == 0:
                    pass
                    # fake_img = sess.run([D_real, D_fake])
                    # for idx, img in enumerate(fake_img[0]):
                    #     img = (img * 255).astype('uint8')
                    #     # img = np.reshape(img, (32, 32, 3))
                    #     img_raw = Image.fromarray(img)
                    #     if not os.path.exists('test/' + str(i)):
                    #         os.mkdir('test/' + str(i))
                    #     img_raw.save('test/' + str(i) + '/' + str(idx) + '.bmp')
                else:
                    errD_fake, errD_real, errG, _, _ = sess.run(
                        [D_fake_loss, D_real_loss, g_loss, d_optim, g_optim])
                    if i % 100 == 0:
                        print(errD_fake, errD_real, errG)
                        print(sess.run([D_real, D_fake]))
