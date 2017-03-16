# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import os
import argparse

import issue_regression
import issue_classification


def classification(args):
    # start to train
    data_name = 'cifar10'

    if args.task == 'train' and args.model is None:
        issue_classification.train.run(data_name, args.net, chkp_path=None)

    # finetune
    elif args.task == 'train' and args.model is not None:
        issue_classification.train.run(data_name, args.net, args.model)

    # test
    elif args.task == 'test' and args.model is not None:
        issue_classification.test.run(data_name, args.net, model_path=args.model)

    # feature
    else:
        raise ValueError('Error task type ', args.task)


def regression(args):
    # start to train
    data_name = 'avec2014'

    if args.task == 'train' and args.model is None:
        issue_regression.train.run(data_name, args.net, chkp_path=None)

    # finetune
    elif args.task == 'train' and args.model is not None:
        issue_regression.train.run(data_name, args.net, args.model)

    # test
    elif args.task == 'test' and args.model is not None:
        issue_regression.test.run(data_name, args.net, model_path=args.model)

    # feature
    else:
        raise ValueError('Error task type ', args.task)


def interface(args):
    """ interface related to command
    """
    # check model
    if isinstance(args.model, str):
        if not os.path.isdir(args.model):
            raise ValueError('Error model path: ', args.model)

    # check net
    if args.net is None:
        raise ValueError('Without input net')

    if args.target == 'regression':
        regression(args)

    if args.target == 'classification':
        classification(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-target', type=str, default='regression', dest='target',
                        help='regression/classification')
    parser.add_argument('-task', type=str, default='train', dest='task',
                        help='train/eval/finetune/feature')
    parser.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder: automatically use newest model')
    parser.add_argument('-net', type=str, default=None, dest='net',
                        help='lenet/cifarnet')
    args, _ = parser.parse_known_args()
    interface(args)
