
import os
import argparse

from issue_regression import regression_train
from issue_regression import regression_test


def interface(args):
    """ interface related to command
    """
    data_name = 'avec2014'

    # check model
    if isinstance(args.model, str):
        if not os.path.isdir(args.model):
            raise ValueError('Error model path: ', args.model)

    # check net
    if args.net is None:
        raise ValueError('Without input net')

    # start to train
    if args.task == 'train' and args.model is None:
        regression_train.train(data_name, args.net, chkp_path=None)

    # finetune
    elif args.task == 'train' and args.model is not None:
        regression_train.train(data_name, args.net, args.model)

    # test
    elif args.task == 'test' and args.model is not None:
        regression_test.test(data_name, args.net, model_path=args.model)

    # feature
    else:
        raise ValueError('Error task type ', args.task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, default='train', dest='task',
                        help='train/eval/finetune/feature')
    parser.add_argument('-model', type=str, default=None, dest='model',
                        help='path to model folder: automatically use newest model')
    parser.add_argument('-net', type=str, default=None, dest='net',
                        help='lenet/cifarnet')
    args, _ = parser.parse_known_args()
    interface(args)
