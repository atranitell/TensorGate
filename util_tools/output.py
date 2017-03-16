# -*- coding: utf-8 -*-
""" updated: 2017/3/16
"""

import sys
# import time


class progressive():

    def __init__(self, min_scale=1.0):
        sys.stdout.write('[TEST] RUN [')
        tab = ''
        self.min_scale = min_scale
        self.num_partion = int(100 / self.min_scale)
        for _ in range(self.num_partion + 4):
            tab += ' '
        sys.stdout.write(tab + ']')
        self.cur = 0

    def add(self):
        self.cur += 1
        if self.cur > self.num_partion:
            return

        for j in range(self.num_partion + 6 - self.cur):
            sys.stdout.write('\b')

        sys.stdout.write('#')

        for m in range(self.num_partion - self.cur):
            sys.stdout.write(' ')

        c = self.cur * self.min_scale
        sys.stdout.write('] %2d%%' % c)

        sys.stdout.flush()
        # time.sleep(0.01)

    def add_float(self, cur, max_cur):
        # print(self.min_scale*self.cur)
        if int(cur * 100 / max_cur) > self.min_scale * self.cur:
            self.add()
        else:
            return

# a = progressive(min_scale=1.0)
# for i in range(105):
#     a.add()


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
              (net_model, dataset.preprocessing_method))
