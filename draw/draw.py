
import re
import matplotlib.pyplot as plt
import math

SN = 5  # smooth num


def parse_log_test(path):
    fp = open(path, 'r')
    tst_info = {}
    _iter = []
    _loss = []
    _mae = []
    _rmse = []
    idx = -1
    phase = False
    for line in fp:
        tst_line = re.findall('\[TEST\] Iter:(.*?),', line)
        if len(tst_line):
            idx = 0
            phase = True
            tst_iter = int(tst_line[0])
            _iter.append(tst_iter)
        if phase is True:
            idx += 1
        if idx == 2:
            idx = -1
            phase = False
            print(line)
            _loss.append(float(re.findall('Loss:(.*?),', line)[0]))
            _mae.append(float(re.findall('mae:(.*?),', line)[0]))
            _rmse.append(float(re.findall('rmse:(.*)', line)[0]))

    tst_info['iter'] = _iter
    tst_info['loss'] = _loss
    tst_info['mae'] = _mae
    tst_info['rmse'] = _rmse

    with open('new_test.txt', 'w') as fw:
        for i, v in enumerate(tst_info['iter']):
            fw.write(str(tst_info['iter'][i]) + ' ' + str(tst_info['loss'][i]) + ' ' +
                     str(tst_info['mae'][i]) + ' ' + str(tst_info['rmse'][i]) + '\n')

    return tst_info


def parse_log(path):
    fp = open(path, 'r')
    trn_info = {}
    _iter = []
    _loss = []
    _mae = []
    _rmse = []
    for line in fp:
        trn_line = re.findall('\[TRAIN\] Iter:(.*?),', line)
        if len(trn_line):
            trn_iter = int(trn_line[0])
            if trn_iter < 100:
                continue
            _iter.append(trn_iter)
            _loss.append(float(re.findall('loss:(.*?),', line)[0]))
            _mae.append(float(re.findall('mae:(.*?),', line)[0]))
            _rmse.append(float(re.findall('rmse:(.*?),', line)[0]))

    trn_info['iter'] = _iter
    trn_info['loss'] = _loss
    trn_info['mae'] = _mae
    trn_info['rmse'] = _rmse

    with open('new.txt', 'w') as fw:
        for i, v in enumerate(trn_info['iter']):
            fw.write(str(trn_info['iter'][i]) + ' ' + str(trn_info['loss'][i]) + ' ' +
                     str(trn_info['mae'][i]) + ' ' + str(trn_info['rmse'][i]) + '\n')

    return trn_info


# num must to be odd number
def smooth(data_tup, num=SN):
    mid = math.floor(num / 2.0)
    for i in range(mid, len(data_tup) - num):
        avg = 0
        for j in range(num):
            avg = avg + data_tup[i + j]
        data_tup[i] = avg / num
    return data_tup


def draw(info):
    plt.plot(info['iter'][:-SN], info['mae'][:-SN], 'b', alpha=0.4)
    plt.plot(info['iter'][:-SN], smooth(info['mae'])[:-SN], 'b')

    plt.plot(info['iter'][:-SN], info['rmse'][:-SN], 'g', alpha=0.4)
    plt.plot(info['iter'][:-SN], smooth(info['rmse'])[:-SN], 'g')

    plt.legend(('mae', 'rmse'))
    plt.grid()
    plt.xlim((0, info['iter'][-1]))
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('cifarnet-avec2014-image')
    plt.show()


r = parse_log_test('log.txt')
r1 = parse_log('log.txt')
draw(r)
