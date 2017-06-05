
import json
import math
import re
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------
#  DRAW DISTRIBUTION OF PP_TRN.TXT
#
# ----------------------------------------
def data_distribution(trn_path, tst_path):
    """ check distribution of trn and test
        trn.txt / tst.txt
    """
    pp = []
    label = []
    test = []
    with open(trn_path, 'r') as fp:
        for line in fp:
            r = line.split(' ')
            pp.append(r[0])
            label.append(int(r[1]))

    with open(tst_path, 'r') as fp:
        for line in fp:
            r = line.split(' ')
            test.append(int(r[1]))

    plt.hist(label, bins=63, rwidth=0.5, alpha=0.5)
    plt.hist(test, bins=63, rwidth=0.5, alpha=0.5)
    plt.xlabel('depression grade')
    plt.ylabel('count')
    plt.title('distribution of degression grade')
    plt.legend(('train', 'test'))
    plt.xticks(np.arange(0, 63, 10.0))
    plt.grid()
    plt.show()


# ----------------------------------------
#  DRAW SINGLE DATA
#
# ----------------------------------------
def smooth(data_tup, num):
    """ K means
    """
    mid = math.floor(num / 2.0)
    for i in range(mid, len(data_tup) - num):
        avg = 0
        for j in range(num):
            avg = avg + data_tup[i + j]
        data_tup[i] = avg / num
    return data_tup


def parse_log_test(path, is_save=False, save_path=None, old=True):
    fp = open(path, 'r')
    tst_info = {}
    _iter = []
    _loss = []
    _mae = []
    _rmse = []
    phase = False
    for line in fp:
        tst_line = re.findall('\[TST\] Iter:(.*?),', line)
        if old:
            if len(tst_line):
                phase = True
                tst_iter = int(tst_line[0])
                _iter.append(tst_iter)

            if phase and len(re.findall('video_mae:(.*?),', line)):
                _mae.append(float(re.findall('video_mae:(.*?),', line)[0]))
                _rmse.append(float(re.findall('video_rmse:(.*)', line)[0]))
                _loss.append(float(re.findall('Loss:(.*?),', line)[0]))
                phase = False
        else:
            if len(tst_line):
                phase = True

            if phase and len(re.findall('video_mae:(.*?),', line)):
                tst_iter = int(tst_line[0])
                _iter.append(tst_iter)
                _mae.append(float(re.findall('video_mae:(.*?),', line)[0]))
                _rmse.append(float(re.findall('video_rmse:(.*).', line)[0]))
                _loss.append(float(re.findall('loss:(.*?),', line)[0]))
                phase = False

    tst_info['path'] = path
    tst_info['iter'] = _iter
    tst_info['loss'] = _loss
    tst_info['mae'] = _mae
    tst_info['rmse'] = _rmse

    print(len(tst_info['iter']))

    if is_save and save_path is not None:
        with open(save_path, 'w') as fw:
            for i, v in enumerate(tst_info['iter']):
                fw.write(str(tst_info['iter'][i]) + ' ' + str(tst_info['loss'][i]) + ' ' +
                         str(tst_info['mae'][i]) + ' ' + str(tst_info['rmse'][i]) + '\n')

    return tst_info


def parse_log_train(path, is_save=False, save_path=None, old=True):
    fp = open(path, 'r')
    trn_info = {}
    _iter = []
    _loss = []
    _mae = []
    _rmse = []
    invl = 1
    for idx, line in enumerate(fp):
        if idx % invl != 0:
            continue
        trn_line = re.findall('\[TRN\] Iter:(.*?),', line)

        if old:
            if len(trn_line):
                trn_iter = int(trn_line[0])
                if trn_iter < 10:
                    continue
                _iter.append(trn_iter)
                _loss.append(float(re.findall('loss:(.*?),', line)[0]))
                _mae.append(float(re.findall('mae:(.*?),', line)[0]))
                _rmse.append(float(re.findall('rmse:(.*?),', line)[0]))
        else:
            if len(trn_line):
                trn_iter = int(trn_line[0])
                if trn_iter < 10:
                    continue
                if len(re.findall('loss:(.*?),', line)) > 0:
                    _iter.append(trn_iter)
                    _loss.append(float(re.findall('loss:(.*?),', line)[0]))
                    _mae.append(float(re.findall('mae:(.*?),', line)[0]))
                    _rmse.append(float(re.findall('rmse:(.*?),', line)[0]))

    trn_info['path'] = path
    trn_info['iter'] = _iter
    trn_info['loss'] = _loss
    trn_info['mae'] = _mae
    trn_info['rmse'] = _rmse

    if is_save and save_path is not None:
        with open(save_path, 'w') as fw:
            for i, v in enumerate(trn_info['iter']):
                fw.write(str(trn_info['iter'][i]) + ' ' + str(trn_info['loss'][i]) + ' ' +
                         str(trn_info['mae'][i]) + ' ' + str(trn_info['rmse'][i]) + '\n')

    return trn_info


def draw_single(info, ylim=(0, 15), k=3):
    """ k stands for smooth points.
    """
    plt.plot(info['iter'], info['mae'], 'b', alpha=0.4)
    p1, = plt.plot(info['iter'], smooth(info['mae'], k), 'b')
    plt.plot(info['iter'], info['rmse'], 'g', alpha=0.4)
    p2, = plt.plot(info['iter'], smooth(info['rmse'], k), 'g')
    plt.legend([p1, p2], ['mae', 'rmse'])
    plt.grid()
    plt.xlim((0, info['iter'][-1]))
    plt.ylim(ylim)
    plt.xlabel('iter')
    plt.ylabel('mae/rmse')
    plt.title(info['path'])
    plt.show()

# TEST
# info = parse_log_test(
#     'avec2014/avec2014_flow_log/2_0.001.txt', True, 'test.txt')
# draw_single(info, (7, 14))


# ----------------------------------------
#  DRAW MULTIPLY DATA
#
# ----------------------------------------
def draw_multiple(filename):
    """ input a json file.
    """
    with open(filename) as fp:
        infos = json.load(fp)

    show_list = []
    for idx in infos['logs']:
        if idx['show'] is True:
            show_list.append(idx)

    for idx in show_list:
        if infos['phase'] == 'train':
            idx['data'] = parse_log_train(idx['path'])
        elif infos['phase'] == 'test':
            idx['data'] = parse_log_test(idx['path'])
        else:
            raise ValueError('Unkonwn type.')

    legends = ()
    for idx in show_list:
        data = idx['data']
        ylabel = infos['ylabel']
        xlabel = infos['xlabel']
        plt.plot(data[xlabel], smooth(data[ylabel], infos['smooth']), alpha=1)
        legends += (idx['legend'],)

    plt.grid()
    plt.legend(legends)
    plt.xlim((0, data['iter'][-1]))
    plt.ylim((infos['ymin'], infos['ymax']))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(infos['title'])
    plt.show()
