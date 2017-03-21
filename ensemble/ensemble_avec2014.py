
import json
import re
import numpy as np
import math


def parse_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


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


def get_ensemble_avg(ensemble, res_list):
    ensemble['avg'] = {}
    for idx, val in enumerate(res_list):
        # acquire first model
        if idx == 0:
            ensemble['avg'] = val
            continue

        for key in val:
            if ensemble['avg'][key][0] != val[key][0]:
                raise ValueError('Error!')
            ensemble['avg'][key][1] += val[key][1]

    for key in ensemble['avg']:
        ensemble['avg'][key][1] /= float(len(res_list))

    ensemble['avg_res'] = get_mae_rmse(ensemble['avg'])


def get_ensemble_value(json_file_path):

    models = parse_json(json_file_path)

    res = {}
    res_list = []
    for idx, val in enumerate(models['list']):
        if val['type'] == 'seq':
            res_dict = get_seq_list(val['path'])
        elif val['type'] == 'img':
            res_dict = get_img_list(val['path'])

        mae, rmse, _ = get_mae_rmse(res_dict)
        res[val['model']] = [val['ensemble'], mae, rmse]

        if val['ensemble'] is True:
            res_list.append((res_dict))

    print('%-20s %5s  %5s  %7s' % ('model name:', 'mae', 'rmse', 'ensemble'))
    for line in res:
        print('%-20s  %-2.3f  %-2.3f  %-7s' % (line, res[line][1], res[line][2], str(res[line][0])))

    # ensemble
    ensemble = {}
    get_ensemble_avg(ensemble, res_list)
    print('\n%-20s  %-2.3f  %-2.3f' %
          ('ensemble_avg', ensemble['avg_res'][0],  ensemble['avg_res'][1]))

get_ensemble_value('list.json')
