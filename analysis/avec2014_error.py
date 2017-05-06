
import os
import math
import re
import numpy as np

""" Input: a log file
    Output: {'class1':[label, logit], ...}
"""


def get_seq_list(path):
    """ For video input
        class1 3 2.53
        class2 4 4.67
        ...
    Return:
        {'class1':[label, logit], ...}
    """
    result = {}
    res_fp = open(path, 'r')
    for line in res_fp:
        r1 = re.findall('frames_flow\\\\(.*)_video', line)
        r2 = re.findall('frames\\\\(.*)_video', line)

        res = r1[0] if len(r1) else r2[0]

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        label = float(label[0])
        logit = float(logit[0])

        result[res] = [label, logit]
    return result


def get_succ_list(path):
    """ For succ video sequence
    """
    res_fp = open(path, 'r')
    res_label = {}
    res_logit = {}
    for line in res_fp:
        r1 = re.findall('frames_flow\\\\(.*)_video', line)
        r2 = re.findall('frames\\\\(.*)_video', line)

        res = r1[0] if len(r1) else r2[0]

        if res.find('frames') >= 0:
            print(res)

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        if res not in res_label:
            res_label[res] = [float(label[0])]
        else:
            res_label[res].append(float(label[0]))

        logit_f = float(logit[0])

        if res not in res_logit:
            res_logit[res] = [logit_f]
        else:
            res_logit[res].append(logit_f)

    # acquire mean
    result = {}
    for idx in res_label:
        label = np.mean(np.array(res_label[idx]))
        logit = np.mean(np.array(res_logit[idx]))
        result[idx] = [label, logit]

    return result


def get_img_list(path):
    """ For image input
        class1 2 3.56
        class1 2 4.32
        class1 2 1.23
        ...
        class2 3 2.11
        ...
    Return:
        Note: the logit will be mean of the value,
          because the label is same value in same class
        {'class1':[label, logit], ...}
    """
    res_fp = open(path, 'r')
    res_label = {}
    res_logit = {}
    for line in res_fp:
        r1 = re.findall('frames_flow\\\(.*?)_video', line)
        r2 = re.findall('frames\\\(.*?)_video', line)

        res = r1[0] if len(r1) else r2[0]

        label = re.findall(' (.*?) ', line)
        logit = re.findall(label[0] + ' (.*)\n', line)

        if res not in res_label:
            res_label[res] = [float(label[0])]
        else:
            res_label[res].append(float(label[0]))

        logit_f = float(logit[0])

        if res not in res_logit:
            res_logit[res] = [logit_f]
        else:
            res_logit[res].append(logit_f)

    # acquire mean
    result = {}
    for idx in res_label:
        label = np.mean(np.array(res_label[idx]))
        logit = np.mean(np.array(res_logit[idx]))
        result[idx] = [label, logit]
    return result


def get_mae_rmse(res):
    """
    Input: a dict
        {'class1':[label value],
         'class2':[label value]}
    """
    mae = 0.0
    rmse = 0.0
    for idx in res:
        mae += abs(res[idx][0] - res[idx][1])
        rmse += math.pow(res[idx][0] - res[idx][1], 2)
    mae = mae / len(res)
    rmse = math.sqrt(rmse / len(res))
    return mae, rmse, len(res)


def get_mae(res):
    mae = 0.0
    for idx in res:
        mae += abs(res[idx][0] - res[idx][1])
    mae = mae / len(res)
    return mae, len(res)


def get_rmse(res):
    rmse = 0.0
    for idx in res:
        rmse += math.pow(res[idx][0] - res[idx][1], 2)
    rmse = math.sqrt(rmse / len(res))
    return rmse, len(res)


def get_accurate_from_file(path, data_type):

    if data_type == 'seq':
        res = get_mae_rmse(get_seq_list(path))
    elif data_type == 'img':
        res = get_mae_rmse(get_img_list(path))
    elif data_type == 'succ':
        res = get_mae_rmse(get_succ_list(path))

    mae, rmse, count = res
    return mae, rmse


def get_all_res_in_fold(path, data_type):
    print(path)

    min_mae = 10000
    min_mae_iter = ''
    min_rmse = 10000
    min_rmse_iter = ''
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)

        if data_type == 'seq':
            res = get_mae_rmse(get_seq_list(filepath))
        elif data_type == 'img':
            res = get_mae_rmse(get_img_list(filepath))
        elif data_type == 'succ':
            res = get_mae_rmse(get_succ_list(filepath))

        if res[0] < min_mae:
            min_mae = res[0]
            min_mae_iter = filename

        if res[1] < min_rmse:
            min_rmse = res[1]
            min_rmse_iter = filename

        print('(%s, %.4f, %.4f, %d)' % (filename, res[0], res[1], res[2]))

    print('min_mae: %.4f in %s' % (min_mae, min_mae_iter))
    print('min_rmse: %.4f in %s' % (min_rmse, min_rmse_iter))
