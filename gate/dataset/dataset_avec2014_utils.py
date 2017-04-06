
import re
import math
import numpy as np


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
        r1 = re.findall('\\\(.*)_video', line)
        r2 = re.findall('\\\(.*)_video', line)

        res = r1[0] if r1[0] else r2[0]

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
        r1 = re.findall('frames_flow\\\(.*)_video', line)
        r2 = re.findall('frames_flow\\\(.*)_video', line)

        res = r1[0] if r1[0] else r2[0]

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
        r1 = re.findall('frames/(.*?)_video', line)
        r2 = re.findall('frames/(.*?)_video', line)

        res = r1[0] if r1[0] else r2[0]

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
    Input: a get_img_list/get_seq_list dict
        {'class1':[label value], ...}
    """
    mae = 0.0
    rmse = 0.0
    for idx in res:
        mae += abs(res[idx][0] - res[idx][1])
        rmse += math.pow(res[idx][0] - res[idx][1], 2)
    mae = mae / len(res)
    rmse = math.sqrt(rmse / len(res))
    return mae, rmse, len(res)


def get_accurate_from_file(path, data_type):

    if data_type == 'seq':
        res = get_mae_rmse(get_seq_list(path))
    elif data_type == 'img':
        res = get_mae_rmse(get_img_list(path))
    elif data_type == 'succ':
        res = get_mae_rmse(get_succ_list(path))

    mae, rmse, _ = res

    return mae, rmse

# print(get_accurate_from_file('gate/dataset/23401.txt', 'succ'))
