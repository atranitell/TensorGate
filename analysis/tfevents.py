
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

""" See utils/draw.py
"""


def get_image_summary(events_fold_path, tags, iter_tag):
    """ values['train/iter']['train/loss'] = v1
        values['train/iter']['train/err'] = v2
        ....
        The operation will overlap old record!
    """
    values = {}
    events_list = []

    for f in os.listdir(events_fold_path):
        if f.find('events.out') == 0:
            events_list.append(os.path.join(events_fold_path, f))
    events_list.sort()

    for event_file in events_list:
        for e in tf.train.summary_iterator(event_file):
            for v in e.summary.value:
                if v.tag == iter_tag:
                    _iter = str(int(v.simple_value)).zfill(8)
                    values[_iter] = {}
                for idx, tag in enumerate(tags):
                    if v.tag == tag:
                        values[_iter][tag] = v.simple_value
    return values


def get_info_data(values, tags, iter_tag):
    """ info{'train/iter'} = [1, 201, 401, ...]
        info{'train/loss'} = [.5, .4, .23, ...]
        they have same length.
    """
    info = {}
    info[iter_tag] = []
    for tag in tags:
        info[tag] = []
    for i in sorted(values):
        info[iter_tag].append(int(i))
        for tag in tags:
            info[tag].append(values[i][tag])
    return info


def get_tags(target, data_type):
    """ get tags
    input:
        target = regression / classification
        data_type = train / test

    return:
        tags: a list of e.g. ['test/mae', 'test/rmse', 'test/loss']
        iter_tag: a str of e.g. 'test/iter'
    """

    if target is 'regression' and data_type is 'train':
        tags = ['train/lr', 'train/err_mae', 'train/err_mse', 'train/loss']
        iter_tag = 'train/iter'

    elif target is 'regression' and data_type is 'test':
        tags = ['test/mae', 'test/rmse', 'test/loss']
        iter_tag = 'test/iter'

    elif target is 'classification' and data_type is 'train':
        tags = ['test/lr', 'test/err', 'test/loss']
        iter_tag = 'train/iter'

    elif target is 'classification' and data_type is 'test':
        tags = ['test/acc', 'test/loss']
        iter_tag = 'test/iter'

    else:
        raise ValueError('Unkonwn!')

    return tags, iter_tag
