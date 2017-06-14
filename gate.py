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

    elif config.target == 'cnn.fuse_cosine.2f':
        import issue.cnn.fuse_cosine_2f as cnn

    else:
        raise ValueError('Unkonwn target type.')

    # if model has value, continue to train, verse vice.
    if config.task == 'train':
        cnn.train(config.dataset, config.model)

    # freeze all weights except extension, and train extension
    elif config.task == 'finetune' and config.model is not None:
        if config.init:
            # exclusions = {'restore': ['net', 'global_step', 'updater'],
            #               'train': ['InceptionResnetV1']}
            exclusions = {'restore': ['updater'],
                          'train': ['net/resnet_v2_50/conv',
                                    'net/resnet_v2_50/block']}
        else:
            exclusions = {'restore': None, 'train': ['InceptionResnetV1']}
        cnn.train(config.dataset, config.model, exclusions)

    # test model
    elif config.task == 'test' and config.model is not None:
        if config.all:
            cnn.pipline(config.dataset, config.model)
        else:
            cnn.test(config.dataset, config.model)

    # validation a model, for specific method.
    #   get a value through val and then test
    elif config.task == 'val' and config.model is not None:
        cnn.val(config.dataset, config.model)

    elif config.task == 'heatmap' and config.model is not None:
        if config.all:
            cnn.pipline(config.dataset, config.model, True)
        else:
            cnn.heatmap(config.dataset, config.model)

    elif config.task == 'extract_feature' and config.model is not None:
        import issue.cnn.extract_feature as extract_feature
        extract_feature.extract_feature(
            config.dataset, config.model, 'PostPool')

    else:
        logger.error('Wrong task setting %s' % str(config.task))


def interface_rnn(config):
    """ interface related to LSTM/RNN/GRU
    """
    if config.target == 'lstm.basic':
        import issue.rnn.classification as rnn
    elif config.target == 'rnn.classification.cnn':
        import issue.rnn.classification_cnn as rnn
    elif config.target == 'rnn.regression.cnn.video':
        import issue.rnn.regression_cnn_video as rnn
    elif config.target == 'rnn.regression.audio':
        import issue.rnn.regression_audio as rnn
    else:
        raise ValueError('Unkonwn target type.')

    if config.task == 'train':
        rnn.train(config.dataset, config.model)
    else:
        raise ValueError('Unkonwn target type.')


def interface_gan(config):
    """ interface related to GAN
    """
    if config.target == 'gan.conditional':
        import issue.gan.cgan as gan
    elif config.target == 'gan.wasserstein':
        import issue.gan.wgan as gan
    else:
        raise ValueError('Unkonwn target type.')

    if config.task == 'train':
        gan.train(config.dataset, config.model)
    elif config.task == 'test':
        gan.test(config.dataset, config.model)
    else:
        raise ValueError('Unkonwn target type.')


def interface_vae(config):
    """ interface related to VAE
    """
    if config.target == 'vae.conditional':
        import issue.vae.conditional_vae as vae
    else:
        raise ValueError('Unkonwn target type.')

    if config.task == 'train':
        vae.train()


def interface(config):
    """ interface related to command
    """
    logger.info(str(config))

    if config.target.find('cnn.') == 0:
        interface_cnn(config)

    elif config.target.find('gan.') == 0:
        interface_gan(config)

    elif config.target.find('vae.') == 0:
        interface_vae(config)

    elif config.target.find('rnn.') == 0:
        interface_rnn(config)

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
    PARSER.add_argument('-all', type=bool, default=False, dest='all')
    ARGS, _ = PARSER.parse_known_args()

    # at least input target, task, dataset
    raise_invalid_input(ARGS.target, ARGS.task, ARGS.dataset)

    # initalize logger
    if ARGS.model is None:
        LOG_PATH = '../_output/' + ARGS.dataset + \
            '_' + ARGS.target + '_' + ARGS.task + \
            datetime.strftime(datetime.now(), '_%y%m%d%H%M%S') + '.txt'
    else:
        LOG_PATH = os.path.join(
            ARGS.model, ARGS.dataset + '_' +
            ARGS.target + '_' + ARGS.task +
            datetime.strftime(datetime.now(), '_%y%m%d%H%M%S') + '.txt')

    # LOG = logger.LOG
    logger.set_filestream(LOG_PATH)
    logger.set_screenstream()

    # output GPU info
    logger.info('SYSTEM WILL RUN ON GPU ' + os.environ["CUDA_VISIBLE_DEVICES"])

    # start
    interface(ARGS)
