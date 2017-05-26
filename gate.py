# -*- coding: utf-8 -*-
""" All interface to access the framework
    Author: Kai JIN
    Updated: 2017/05/19
"""
import os
import argparse
import sys
from datetime import datetime

# hidden device output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from gate.utils.logger import logger

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


def interface_cnn(config):
    # choose different task
    if config.target == 'cnn.regression':
        import issue.cnn.regression as cnn

    elif config.target == 'cnn.regression.4view':
        import issue.cnn.regression_4view as cnn

    elif config.target == 'cnn.classification':
        import issue.cnn.classification as cnn

    elif config.target == 'cnn.fuse_cosine':
        import issue.cnn.fuse_cosine as cnn

    elif config.target == 'cnn.fuse_cosine.mw':
        import issue.cnn.fuse_cosine_multiway as cnn

    else:
        raise ValueError('Unkonwn target type.')

    # differnt method to finish task
    if config.task == 'train' and config.model is None:
        cnn.train(config.dataset)

    # continue to train
    elif config.task == 'train' and config.model is not None:
        cnn.train(config.dataset, config.model)

    # freeze all weights except extension, and train extension
    elif config.task == 'finetune' and config.model is not None:
        if config.init:
            exclusions = {'restore': ['net1', 'net2', 'global_step', 'updater'],
                          'train': ['InceptionResnetV1']}
        else:
            exclusions = {'restore': None, 'train': ['InceptionResnetV1']}
        cnn.train(config.dataset, config.model, exclusions)

    # test model
    elif config.task == 'test' and config.model is not None:
        cnn.test(config.dataset, config.model)

    # validation a model, for specific method.
    #   get a value through val and then test
    elif config.task == 'val' and config.model is not None:
        cnn.val(config.dataset, config.model)

    elif config.task == 'heatmap' and config.model is not None:
        cnn.heatmap(config.dataset, config.model)

    elif config.task == 'extract_feature' and config.model is not None:
        import issue.cnn.extract_feature as extract_feature
        extract_feature.extract_feature(
            config.dataset, config.model, 'PostPool')

    else:
        logger.error('Wrong task setting %s' % str(config.task))


def interface(config):
    """ interface related to command
    """
    logger.info(str(config))

    if config.target.find('cnn') == 0:
        interface_cnn(config)

    elif config.target == 'cgan':
        pass

    elif config.target == 'vae':
        pass

    else:
        logger.error('Wrong target setting %s' % str(config.target))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-target', type=str, default=None, dest='target',
                        help='cnn/cgan/vae.')
    PARSER.add_argument('-task', type=str, default=None, dest='task',
                        help='train/val/heatmap/extract_feature.')
    PARSER.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder.')
    PARSER.add_argument('-dataset', type=str, default=None, dest='dataset',
                        help='defined in dataset/factory.')
    PARSER.add_argument('-init', type=bool, default=True, dest='init')
    ARGS, _ = PARSER.parse_known_args()

    # at least input target, task, dataset
    raise_invalid_input(ARGS.target, ARGS.task, ARGS.dataset)

    # initalize logger
    LOG_PATH = '../_output/' + ARGS.dataset + \
        '_' + ARGS.target + '_' + ARGS.task + \
        datetime.strftime(datetime.now(), '_%y%m%d%H%M%S') + '.txt'

    # LOG = logger.LOG
    logger.set_filestream(LOG_PATH)
    logger.set_screenstream()

    # output GPU info
    logger.info('SYSTEM WILL RUN ON GPU ' + os.environ["CUDA_VISIBLE_DEVICES"])

    # start
    interface(ARGS)
