# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

# from data import datasets_factory
# from data import dataset_tfrecords

# dataset = datasets_factory.get_dataset('avec2014', 'train')
# dataset_tfrecord = dataset_tfrecords.tfrecord()

# # dataset_tfrecord.process(dataset
# dataset_tfrecord.read_from_tfrecord(dataset.data_path)

# import issue_regression.train_fuse

# issue_regression.train_fuse.run('avec2014', 'cifarnet', chkp_path=None)


def foo(*args):
    for arg in args:
        if arg is None:
            raise ValueError('Input is None type, Please check again.')



def a(k1, k2, k3):
    pass

def f1(func):
    print(func.__name__)

f1(a(1,2,3))
