
import os
import numpy as np
import scipy.stats as stat
import issue.image.kinface_utils as kinface_utils


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
        # dis, _ = stat.spearmanr(p/norm_p, c/norm_c)
        # dis = np.linalg.norm(abs(p-c))
        # dis = np.linalg.norm(p/norm_p-c/norm_c)
        rw_f.write(line[0:-1] + ' ' + str(dis) + '\n')
        i += 1


def get_result(p_fp, c_fp, r_fp, rw_fp):
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

    res = kinface_utils.parse_result(rw_fp)
    
    # print(len(res))
    _best_err, _best_val = kinface_utils.get_best_threshold(res)
    return _best_val


def get_test(rw_fp, val):
    acc = kinface_utils.get_kinface_error(rw_fp, val)
    print(rw_fp, val, acc)
    return acc
