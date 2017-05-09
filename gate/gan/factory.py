# -*- coding: utf-8 -*-
""" updated: 2017/05/09
"""

import tensorflow as tf
from gate.gan import condition_gan
from gate.gan import wassertein_gan

model_map = {
    'cgan': condition_gan.CGAN,
    'wgan': wassertein_gan.WGAN
}


def get_model(name, data_type, dataset, name_scope=''):
    """
    """
    is_training = True if data_type == 'train' else False
    if name not in model_map:
        raise ValueError('Unknown data_type %s' % name)
    with tf.variable_scope('net' + name_scope):
        return model_map[name](dataset, is_training)
