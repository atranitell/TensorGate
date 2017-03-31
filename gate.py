# -*- coding: utf-8 -*-
""" All interface to access the framework
"""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import issue.image.regression as img_regression
import issue.image.regression_fuse as img_regression_fuse
import issue.image.classification as img_classification


def raise_invalid_input(*config):
    """ Check input if none
    """
    for arg in config:
        if arg is None:
            raise ValueError('Input is None type, Please check again.')


def classification_for_image(config):
    """ classification for image
    """

    if config.task == 'train' and config.model is None:
        raise_invalid_input(config.dataset, config.net)
        img_classification.train(config.dataset, config.net)

    # continue to train
    elif config.task == 'train' and config.model is not None:
        raise_invalid_input(config.dataset, config.net, config.model)
        img_classification.train(config.dataset, config.net, config.model)

    # test
    elif config.task == 'test':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_classification.test(config.dataset, config.net, config.model)

    else:
        raise ValueError('Error task type %s' % config.task)


def regression_fuse_for_image(config):
    """ Regression for image
    """
    config.dataset = 'avec2014_fuse_16f'
    config.net = 'lightnet'

    # train from start
    if config.task == 'train' and config.model is None:
        raise_invalid_input(config.dataset, config.net)
        img_regression_fuse.train(config.dataset, config.net)

    # train continue
    elif config.task == 'train' and config.model is not None:
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression_fuse.train(config.dataset, config.net, config.model)

    # test all samples once
    elif config.task == 'test':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression_fuse.test(config.dataset, config.net, config.model)

    # train from saved model and fixed some layers
    elif config.task == 'finetune':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression_fuse.train(config.dataset, config.net, config.model,
                                  exclusions=['cifarnet/fc3', 'cifarnet/fc4'])
    else:
        raise ValueError('Error task type: %s' % config.task)


def regression_for_image(config):
    """ Regression for image
    """
    # train from start
    if config.task == 'train' and config.model is None:
        raise_invalid_input(config.dataset, config.net)
        img_regression.train(config.dataset, config.net)

    # train continue
    elif config.task == 'train' and config.model is not None:
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.train(config.dataset, config.net, config.model)

    # test all samples once
    elif config.task == 'test':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.test(config.dataset, config.net, config.model)

    # train from saved model and fixed some layers
    elif config.task == 'finetune':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.train(config.dataset, config.net, config.model,
                             exclusions=['cifarnet/fc3', 'cifarnet/fc4'])
    else:
        raise ValueError('Error task type: %s' % config.task)


def interface(config):
    """ interface related to command
    """
    print('[INFO] ', config)

    if config.target == 'regression':
        regression_for_image(config)

    if config.target == 'classification':
        classification_for_image(config)

    if config.target == 'regression_fuse':
        regression_fuse_for_image(config)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-target', type=str, default='regression', dest='target',
                        help='regression/classification/regression_fuse')
    PARSER.add_argument('-task', type=str, default='train', dest='task',
                        help='train/test/finetune/feature')
    PARSER.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder: automatically use newest model')
    PARSER.add_argument('-net', type=str, default='lightnet', dest='net',
                        help='lenet/cifarnet')
    PARSER.add_argument('-dataset', type=str, default=None, dest='dataset',
                        help='avec2014/cifar10')
    ARGS, _ = PARSER.parse_known_args()
    interface(ARGS)
