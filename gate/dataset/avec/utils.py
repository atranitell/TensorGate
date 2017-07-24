
import os
import math
import re
import numpy as np

""" Input: a log file
    Output: {'class1':[label, logit], ...}
"""

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


def get_accurate_from_file(path, data_type=None):
    res = get_mae_rmse(get_img_list(path))
    mae, rmse, count = res
    return mae, rmse


def get_all_res_in_fold(path):
    print(path)
    min_mae = 10000
    min_mae_iter = ''
    min_rmse = 10000
    min_rmse_iter = ''
    for filename in sorted(os.listdir(path)):
        filepath = os.path.join(path, filename)

        res = get_mae_rmse(get_img_list(filepath))

        if res[0] < min_mae:
            min_mae = res[0]
            min_mae_iter = filename

        if res[1] < min_rmse:
            min_rmse = res[1]
            min_rmse_iter = filename

        print('(%s, %.4f, %.4f, %d)' % (filename, res[0], res[1], res[2]))

    print('min_mae: %.4f in %s' % (min_mae, min_mae_iter))
    print('min_rmse: %.4f in %s' % (min_rmse, min_rmse_iter))


""" FIND BEST MARGIN
"""


def get_result_entry(filepath):
    """ ps: the result file should according to time order.
    """
    data = {}
    with open(filepath) as fp:
        for line in fp:
            res = line.split(' ')
            if res[0] not in data:
                data[res[0]] = []
            data[res[0]].append(line)
    return data


def get_accurate_with_margin(filepath, s1, s2):
    """ return sub-entries result.
    """
    entries = get_result_entry(filepath)
    tmp_file = '_tmp_margin_file_' + str(np.random.uniform(0, 10000000)) + '.txt'
    fw = open(tmp_file, 'w')
    for i in entries:
        size = len(entries[i])
        clip1 = int(size * s1)
        clip2 = int(size * s2)
        # print(k1, k2)
        for j in range(clip1, size - clip2):
            fw.write(entries[i][j])
    fw.close()
    mae, rmse = get_accurate_from_file(tmp_file)
    os.remove(tmp_file)
    return mae, rmse


def find_best_margin(filepath, um=0.05):
    """ acquire a best margin on training set.
        PS: only used in training set.
    """
    min_mae, min_rmse = 100, 100
    min_s1, min_s2 = 0, 0
    upper = int(0.5 / um)
    for k1 in range(0, upper):
        s1 = um * k1
        for k2 in range(0, upper):
            s2 = um * k2
            mae, rmse = get_accurate_with_margin(filepath, s1, s2)
            if mae < min_mae:
                min_mae, min_rmse = mae, rmse
                min_s1, min_s2 = s1, s2
    # print('Best result: ', min_s1, min_s2, min_data)
    return min_s1, min_s2, min_mae, min_rmse
