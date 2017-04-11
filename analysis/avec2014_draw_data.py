
import matplotlib.pyplot as plt
import numpy as np


def dist_data():
    pp = []
    label = []
    test = []
    with open('analysis/avec2014/pp_trn.txt', 'r') as fp:
        for line in fp:
            r = line.split(' ')
            pp.append(r[0])
            label.append(int(r[1]))

    with open('analysis/avec2014/pp_tst.txt', 'r') as fp:
        for line in fp:
            r = line.split(' ')
            test.append(int(r[1]))

    # print(pp)
    # print(label)
    plt.hist(label, bins=63, rwidth=0.5, alpha=0.5)
    plt.hist(test, bins=63, rwidth=0.5, alpha=0.5)
    plt.xlabel('depression grade')
    plt.ylabel('count')
    plt.title('distribution of degression grade')
    plt.legend(('train', 'test'))
    plt.xticks(np.arange(0, 63, 10.0))
    plt.grid()
    plt.show()

dist_data()