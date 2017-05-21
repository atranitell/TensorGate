
import os
import numpy as np
import scipy.stats as stat


def get_result(res, val):
    correct = 0.0
    for i in range(len(res)):
        if res[i][2] == 1 and res[i][3] >= val:
            correct += 1
        if res[i][2] == -1 and res[i][3] < val:
            correct += 1
        if res[i][2] == 0 and res[i][3] < val:
            correct += 1
    return correct / len(res)


def get_best_threshold(res):
    size = len(res)
    _best_val = 0
    _best_err = 0
    for i in range(size):
        _cur_val = res[i][3]
        _cur_err = get_result(res, _cur_val)
        if _cur_err > _best_err:
            _best_err = _cur_err
            _best_val = _cur_val
    # print('Best Threshold:%.4f, Best Error:%.4f' % (_best_val, _best_err))
    return _best_err, _best_val


def parse_result(filepath):
    result = []
    with open(filepath) as fp:
        for line in fp:
            r = line.split(' ')
            result.append((r[0], r[1], int(r[2]), float(r[3])))
    return result


def get_kinface_error(filepath, val=None):
    if val is None:
        res = parse_result(filepath)
        return get_best_threshold(res)
    else:
        res = parse_result(filepath)
        return get_result(res, val)

def parse_data(fp):
    res = []
    for line in fp:
        r = line.split(' ')
        val = []
        for i in range(len(r) - 1):
            val.append(float(r[i]))
        res.append(val)
    return res


def distance(p_res, c_res, r_f, rw_f):
    # for i in range(len(p_res)):
    i = 0
    for line in r_f:
        p = np.array(p_res[i])
        c = np.array(c_res[i])
        norm_p = np.linalg.norm(p)
        norm_c = np.linalg.norm(c)
        dis = (np.dot(p, c)) / (norm_c * norm_p)
        rw_f.write(line[0:-2] + ' ' + str(dis) + '\n')
        i += 1


def result(p_fp, c_fp, r_fp, rw_fp):
    p_f = open(p_fp, 'r')
    c_f = open(c_fp, 'r')
    r_f = open(r_fp, 'r')
    rw_f = open(rw_fp, 'w')

    # p_f = open('fd_val_2_f.txt', 'r')
    # c_f = open('fd_val_2_d.txt', 'r')
    # r_f = open('fd_val_2.txt', 'r')
    # rw_f = open('fd_val_2_r.txt', 'w')

    p_res = parse_data(p_f)
    c_res = parse_data(c_f)

    distance(p_res, c_res, r_f, rw_f)
    # print(np.linalg.norm([1, 1]))
    rw_f.close()

    res = parse_result(rw_fp)
    # print(len(res))
    _best_err, _best_val = get_best_threshold(res)
    return _best_val


def test(rw_fp, val):
    acc = get_kinface_error(rw_fp, val)
    print(rw_fp, val, acc)
    return acc