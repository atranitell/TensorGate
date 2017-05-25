""" Generate heatmap
"""
import cv2
from matplotlib import pyplot as plt


def single_map(path, data, weight, raw_h, raw_w, save_path=''):
    """ a uniform IO for generating a single heatmap
    Input
        ImagePath: a str of path to load image
        data: (C, kernelsize, kernelsize)
        weights: (C, )
        raw_h: height of Image
        raw_w: width of Image
    """
    # dim
    channels = data.shape[0]

    conv_img = data[0] * weight[0]
    for _c in range(1, channels):
        conv_img += data[_c] * weight[_c]
    conv_img = cv2.resize(conv_img, (raw_h, raw_w))

    # get raw image
    src = cv2.imread(path)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # show
    plt.close('all')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.imshow(src)
    plt.imshow(conv_img, cmap='jet', alpha=0.4, interpolation='nearest')

    # save
    plt.savefig(save_path, bbox_inches='tight')
