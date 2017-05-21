
import numpy as np


def scale(res):
    mean = np.mean(res)
    res = res - mean
    # for i in range(len(res)):
    #     res[i] = (max(res) - res[i]) / (max(res) - min(res))
    return res


def get_data(filename):
    res = {}
    with open(filename, 'r') as fp:
        res[0] = []
        res[1] = []
        res[2] = []
        res[3] = []
        for line in fp:
            r = line.split(' ')
            res[0].append(r[0])
            res[1].append(r[1])
            res[2].append(r[2])
            res[3].append(float(r[3]))
        res[3] = scale(res[3])
    return res


def merge(res1, res2, alpha=0.5):
    a1 = alpha
    a2 = 1 - a1
    res = {}
    res[0] = res1[0]
    res[1] = res1[1]
    res[2] = res1[2]
    res[3] = []
    for i in range(len(res1[3])):
        res[3].append((a1 * res1[3][i] + a2 * res2[3][i]))
    return res


def write(res, filename):
    with open(filename, 'w') as fw:
        for idx in range(len(res[0])):
            fw.write(
                res[0][idx] + ' ' + res[1][idx] + ' ' +
                res[2][idx] + ' ' + str(res[3][idx]) + '\n')


import kinface_distance


def voted_ensemble():
    res_trn = []
    res_val = []
    for i in range(1, 6):
        res1 = get_data('1/train_' + str(i) + '_r.txt')
        res3 = get_data('2/train_' + str(i) + '_r.txt')

        best_alpha, best_val, best_acc = 0, 0, 0
        interval = 500
        for idx in range(interval):
            alpha = idx * 1.0 / interval
            res = merge(res1, res3, alpha)
            write(res, 'train_' + str(i) + '.txt')
            acc, val = kinface_distance.get_kinface_error(
                'train_' + str(i) + '.txt')
            if acc > best_acc:
                best_acc = acc
                best_val = val
                best_alpha = alpha

        res1 = get_data('1/val_' + str(i) + '_r.txt')
        res3 = get_data('2/val_' + str(i) + '_r.txt')

        res = merge(res1, res3, best_alpha)
        write(res, 'val_' + str(i) + '.txt')

        test_res = kinface_distance.get_kinface_error(
            'val_' + str(i) + '.txt', best_val)

        res_val.append(test_res)
        res_trn.append(best_acc)
        print(best_val, best_acc, test_res)

    print(res_trn)
    print(res_val)


def avg_ensemble():
    res_trn = []
    res_val = []
    for i in range(1, 6):
        res1 = get_data('1/train_' + str(i) + '_r.txt')
        res3 = get_data('2/train_' + str(i) + '_r.txt')

        res = merge(res1, res3)
        write(res, 'train_' + str(i) + '.txt')

        res, val = kinface_distance.get_kinface_error(
            'train_' + str(i) + '.txt')
        res_trn.append(res)

        res1 = get_data('1/val_' + str(i) + '_r.txt')
        res3 = get_data('2/val_' + str(i) + '_r.txt')

        res = merge(res1, res3)
        write(res, 'val_' + str(i) + '.txt')

        res = kinface_distance.get_kinface_error('val_' + str(i) + '.txt', val)
        res_val.append(res)

    print(res_trn)
    print(res_val)

voted_ensemble()

# avg_ensemble()
