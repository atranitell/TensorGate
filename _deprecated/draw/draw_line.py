# -*- coding: utf-8 -*-
""" offer a set of draw curve method.
    Updated: 2017/06/11
"""
import os
import matplotlib.pyplot as plt
import dataPaser
import utils


def basic_line_chart(config):
    """ config is a dict including:
    1. legend
    2. path: path to log
    3. phase: '[TRN]', '[TST]', '[VAL]'
    4. type: 'loss', 'err', 'mae', etc.
    5. invl: default to 1
    6. y_min, y_max, x_min, x_max: default to None
    7. smooth: default to 1, average with multi-points.
    8. show: default to true, to draw on the figure.
    """
    # initail param
    plt.close('all')
    cfg_fig = config['figure']
    cfg_data = config['data']

    def fill(key, default, cfg=cfg_fig):
        return cfg[key] if key in cfg else default

    # step1: parse data
    for dt in cfg_data:
        # control showing on figure
        if dt['show'] is False:
            continue
        # parse data from file
        res = dataPaser.bigram(dt['path'], dt['phase'], dt['key'])
        # downsampling
        if dt['invl'] > 1:
            res = utils.downsampling_bigram(res, dt['invl'])
        # smooth curve
        if dt['smooth'] > 1:
            res[dt['key']] = utils.smooth(res[dt['key']], dt['smooth'])
        plt.plot(res['iter'], res[dt['key']],
                 label=fill('legend', None, dt), alpha=0.8)
        # save data
        if 'save_data' in dt:
            utils.write_to_text(res, ['iter', dt['key']], dt['save'])

    # step2: config figure
    plt.grid()

    # label
    plt.title(fill('title', 'Line chart'))
    plt.xlabel(fill('xlabel', 'iter'))
    plt.ylabel(fill('ylabel', 'value'))

    # lim
    plt.xlim(xmin=fill('xmin', 0))
    plt.xlim(xmax=fill('xmax', None))
    plt.ylim(ymin=fill('ymin', None))
    plt.ylim(ymax=fill('ymax', None))

    # show legend
    plt.legend()

    # save
    if 'save_fig' in cfg_fig:
        plt.savefig(cfg_fig['save_fig'])

    # plt show
    plt.show()
