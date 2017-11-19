
import math
import matplotlib.pyplot as plt

""" draw a line:
e.g.
    # get specify data
    tags, iter_tag = tfevents.get_tags('regression', 'train')

    # parse data
    values = tfevents.get_image_summary(
        '_output/cifar_train_201703251903', tags, iter_tag)

    # transfer data to draw
    info = tfevents.get_info_data(values, tags, iter_tag)

    # draw data
    draw.draw(info, tags, iter_tag)
"""


def smooth(data_tup, num=3):
    mid = math.floor(num / 2.0)
    for i in range(mid, len(data_tup) - num):
        avg = 0
        for j in range(num):
            avg = avg + data_tup[i + j]
        data_tup[i] = avg / num
    return data_tup


def draw(info, tag, iter_tag):
    plt.plot(info[iter_tag], info[tag], 'r', alpha=0.3)
    plt.plot(info[iter_tag], smooth(info[tag]), 'b', alpha=0.6)
    plt.grid()
    plt.xlim((0, info[iter_tag][-1]))
    plt.xlabel('iter')
    plt.ylabel(tag)
    plt.title('')
    plt.show()
