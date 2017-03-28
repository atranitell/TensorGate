# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""


def print_basic_information(dataset, net_model=None):
    phase = '[' + dataset.data_type.upper() + '] '
    # LOG
    _log = phase + 'Test invl:%d, chkp invl:%d, summary invl:%d, device:%s'
    print(_log % (dataset.log.test_interval,
                  dataset.log.save_model_iter,
                  dataset.log.save_summaries_iter,
                  dataset.device))
    # COMMON
    _common = phase + 'Total num:%d, batch size:%d, height:%d, width:%d'
    print(_common % (dataset.total_num,
                     dataset.batch_size,
                     dataset.output_height,
                     dataset.output_width))

    # path
    print(phase + 'Data path:%s' % dataset.data_path)

    if dataset.data_type == 'train':
        print(phase + 'Model running in ' + dataset.log.train_dir)

    if dataset.data_type == 'test':
        print(phase + 'Model running in ' + dataset.log.test_dir)

    # net_model
    if net_model is not None:
        print(phase + 'Using net model:%s, preprocessing method: %s' %
              (net_model, dataset.preprocessing_method1))
