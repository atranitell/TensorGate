# -*- coding: utf-8 -*-
""" All interface to access the framework
    Author: Kai JIN
    Updated: 2017/05/19
"""
import os
import argparse
import sys
import logging
from datetime import datetime

# hidden device output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# allocate GPU to sepcify device
gpu_cluster = ['0', '1', '2', '3']
gpu_id = '0' if sys.argv[1] not in gpu_cluster else sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


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
                                  exclusions=['fully_connected', 'logits'])
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

    elif config.task == 'test_all':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.test_all(config.dataset, config.net, config.model)

    elif config.task == 'test_heatmap':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.test_heatmap(config.dataset, config.net, config.model)

    elif config.task == 'test_all_heatmap':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.test_all(config.dataset, config.net, config.model, True)

    # train from saved model and fixed some layers
    elif config.task == 'finetune':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression.train(config.dataset, config.net, config.model,
                             exclusions=['net/vgg_16/fc7', 'net/vgg_16/fc8'])
    else:
        raise ValueError('Error task type: %s' % config.task)


def regression_share_for_image(config):
    # train from start
    if config.task == 'train' and config.model is None:
        raise_invalid_input(config.dataset, config.net)
        img_regression_share.train(config.dataset, config.net)

    # train continue
    elif config.task == 'train' and config.model is not None:
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression_share.train(config.dataset, config.net, config.model)

    # test all samples once
    elif config.task == 'test':
        raise_invalid_input(config.dataset, config.net, config.model)
        img_regression_share.test(config.dataset, config.net, config.model)


def interface_cnn(config):
    import issue.cnn as cnn

    if config.task == 'train' and config.model is None:
        cnn.train.run(config.dataset, config.net)

    elif config.task == 'train' and config.model is not None:
        cnn.train.run(config.dataset, config.net, config.model)

    elif config.task == 'val' and config.model is not None:
        cnn.val.run(config.dataset, config.net, config.model)

    elif config.task == 'heatmap' and config.model is not None:
        cnn.heatmap.run(config.dataset, config.net, config.model)

    elif config.task == 'extract_feature' and config.model is not None:
        cnn.extract_feature.run(config.dataset, config.net, config.model)

    else:
        logging.error('Wrong task setting %s' % str(config.task))


def interface_cgan(config):
    pass


def interface_vae(config):
    pass


def interface(config):
    """ interface related to command
    """
    logging.info(str(config) + str(123))
    if config.target == 'cnn':
        interface_cnn(config)

    elif config.target == 'cgan':
        interface_cgan(config)

    elif config.target == 'vae':
        interface_vae(config)

    else:
        logging.error('Wrong target setting %s' % str(config.target))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-target', type=str, default=None, dest='target',
                        help='cnn/cgan/vae.')
    PARSER.add_argument('-task', type=str, default=None, dest='task',
                        help='train/val/heatmap/extract_feature.')
    PARSER.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder.')
    PARSER.add_argument('-net', type=str, default=None, dest='net',
                        help='defined in net/factory.')
    PARSER.add_argument('-dataset', type=str, default=None, dest='dataset',
                        help='defined in dataset/factory.')
    ARGS, _ = PARSER.parse_known_args()

    # at least input target, task, dataset
    raise_invalid_input(ARGS.target, ARGS.task, ARGS.dataset)

    # initalize logger
    LOG_NAME = '../_output/' + ARGS.dataset + \
        '_' + ARGS.target + '_' + ARGS.task + \
        datetime.strftime(datetime.now(), '_%Y%m%d_%H%M%S') + '.txt'
    logging.basicConfig(
        level=logging.INFO, filename=LOG_NAME, filemode='w',
        datefmt='%y.%m.%d %H:%M:%S',
        format='[%(asctime)s] [%(levelname)s] %(message)s')

    # output GPU info
    logging.info('SYSTEM WILL RUN ON GPU ' + os.environ["CUDA_VISIBLE_DEVICES"])

    # start
    interface(ARGS)
