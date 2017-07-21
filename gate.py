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

# allocate GPU to sepcify device
gpu_cluster = ['0', '1', '2', '3']
gpu_id = '0' if sys.argv[1] not in gpu_cluster else sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

from gate.utils.logger import logger
from gate.utils import check
from issue.pipline import pipline
from issue.extract_feature import extract_feature


def task_train(config, target):
    """ if model is not None, it will continue to train from this model.
    """
    target.train(config.dataset, config.model)


def task_finetune(config, target):
    """ 
    Params exclusions['restore'] will be excluded to load in model
    Params exclusions['train] will be excluded to train,
        it means that the params will be freezed.
        and the rest of params will be trained.
    """
    check.raise_none_param(config.model)
    if config.init:
        # exclusions = {'restore': ['net', 'global_step', 'updater'],
        #               'train': ['InceptionResnetV1']}
        exclusions = {'restore': ['updater'],
                      'train': ['net/resnet_v2_50/conv',
                                'net/resnet_v2_50/block']}
    else:
        exclusions = {'restore': None, 'train': ['InceptionResnetV1']}
    target.train(config.dataset, config.model, exclusions)


def task_test(config, target):
    check.raise_none_param(config.model)
    if config.all:
        # NOTE: has not been tested.
        pipline(config.dataset, config.model, fn=target.test)
        # cnn.pipline(config.dataset, config.model)
    else:
        target.test(config.dataset, config.model)


def task_heatmap(config, target):
    check.raise_none_param(config.model)
    if config.all:
        # NOTE: has not been tested.
        pipline(config.dataset, config.model, fn=target.heatmap)
        # cnn.pipline(config.dataset, config.model, True)
    else:
        target.heatmap(config.dataset, config.model)


def task_val(config, target):
    check.raise_none_param(config.model)
    target.val(config.dataset, config.model)


def task_extract_feature(config, target):
    check.raise_none_param(config.model)
    if config.all:
        pipline(config.dataset, config.model,
                fn=extract_feature, layer_name='PostPool')
    else:
        extract_feature(config.dataset, config.model, 'PostPool')


def interface_task(config, target):
    """ A set of standard interface.
        If target has been not realized, it will raise a error.
    """
    if config.task == 'train':
        task_train(config, target)

    elif config.task == 'finetune':
        task_finetune(config, target)

    elif config.task == 'test':
        task_test(config, target)

    elif config.task == 'val':
        task_val(config, target)

    elif config.task == 'heatmap':
        task_heatmap(config, target)

    elif config.task == 'extract_feature':
        task_extract_feature(config, target)

    else:
        logger.error('Wrong task setting %s' % str(config.task))


def interface_target(config):
    """ choose different issue.
    """
    target = None

    """ CNN """
    if config.target == 'cnn.regression':
        import issue.cnn.regression as target
    elif config.target == 'cnn.regression.4view':
        import issue.cnn.regression_4view as target
    elif config.target == 'cnn.regression.audio':
        import issue.cnn.regression_audio as target
    elif config.target == 'cnn.classification':
        import issue.cnn.classification as target
    elif config.target == 'cnn.fuse.cosine':
        import issue.cnn.fuse_cosine as target
    elif config.target == 'cnn.fuse.cosine.5view.gc':
        import issue.cnn.fuse_cosine_5view_gc as target

    """ RNN """
    if config.target == 'lstm.basic':
        import issue.rnn.classification as target
    elif config.target == 'rnn.classification.cnn':
        import issue.rnn.classification_cnn as target
    elif config.target == 'rnn.regression.cnn.video':
        import issue.rnn.regression_cnn_video as target
    elif config.target == 'rnn.regression.audio':
        import issue.rnn.regression_audio as target

    """ GAN """
    if config.target == 'gan.conditional':
        import issue.gan.cgan as target
    elif config.target == 'gan.wasserstein':
        import issue.gan.wgan as target
    elif config.target == 'gan.encoder':
        import issue.gan.egan as target

    """ VAE """
    if config.target == 'vae.conditional':
        import issue.vae.conditional_vae as target

    if target is None:
        raise ValueError('Unkonwn target type.')

    return target


def interface(config):
    """ interface related to command
    """
    logger.info(str(config))
    target = interface_target(config)
    interface_task(config, target)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-target', type=str, default=None, dest='target')
    PARSER.add_argument('-task', type=str, default='train', dest='task')
    PARSER.add_argument('-model', type=str, default=None, dest='model')
    PARSER.add_argument('-dataset', type=str, default=None, dest='dataset')
    PARSER.add_argument('-init', type=bool, default=True, dest='init')
    PARSER.add_argument('-all', type=bool, default=False, dest='all')
    ARGS, _ = PARSER.parse_known_args()

    # at least input target, task, dataset
    check.raise_none_param(ARGS.target, ARGS.task, ARGS.dataset)

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
