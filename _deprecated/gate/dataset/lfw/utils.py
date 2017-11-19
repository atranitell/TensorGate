# -*- coding: utf-8 -*-
""" updated: 2017/07/06
"""

# from scipy import spatial
import numpy as np
import sys


def get_cosine_dist(feat1, feat2):
    p = np.array(feat1)
    c = np.array(feat2)
    norm_p = np.linalg.norm(p)
    norm_c = np.linalg.norm(c)
    dis = (np.dot(p, c)) / (norm_c * norm_p)
    # dis = spatial.distance.cosine(feat1, feat2)
    return 1 - dis


def get_dist(feats):
    dist = []
    for i in range(0, len(feats), 2):
        dist.append(get_cosine_dist(feats[i], feats[i + 1]))
    return np.array(dist)


def get_error(data, score):
    count = 0.
    for i in range(len(data[0])):
        if data[0][i] < score and data[1][i] == 1:
            count += 1
        if data[0][i] >= score and data[1][i] == 0:
            count += 1
    return count / (len(data[0]))


def get_test_error(trn_feats, tst_feats):
    """ (300 positive + 300 negtive) * 9 -> trn_feats
        (300 positive + 300 negtive) * 1 -> tst_feats
    """
    res = []

    label = np.append(np.ones(300, dtype=np.int32),
                      np.zeros(300, dtype=np.int32))
    tst_label = label
    trn_label = label
    for i in range(8):
        trn_label = np.append(trn_label, label)

    trn_dist = get_dist(trn_feats)
    tst_dist = get_dist(tst_feats)

    trn = np.row_stack((trn_dist, trn_label))
    tst = np.row_stack((tst_dist, tst_label))

    trn = trn[:, np.argsort(trn[0, :])]
    tst = tst[:, np.argsort(tst[0, :])]

    best_score, best_err = 0., 0.
    for score in trn[0]:
        err = get_error(trn, score)
        if err > best_err:
            best_err = err
            best_score = score
        # print(score, err)
    print(best_score, best_err)
    print(get_error(tst, best_score))


def test_error(data_path):
    # load data
    feats = np.load(data_path)
    feats_shape = feats.shape
    if len(feats_shape) == 4:
        feats = np.reshape(feats, (feats_shape[0], feats_shape[3]))

    # divide to 10 fold
    fold = {}
    res = 0
    for idx in range(10):
        tst_start = idx * 1200
        tst_end = (idx + 1) * 1200
        tst_set = feats[tst_start:tst_end]
        trn_set = np.row_stack((feats[0:tst_start], feats[tst_end:]))
        get_test_error(trn_set, tst_set)

if __name__ == '__main__':
    data_path = sys.argv[1]
    test_error(data_path)
