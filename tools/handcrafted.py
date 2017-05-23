# -*- coding: utf-8 -*-
""" updated: 2017/5/23
"""
import numpy as np
import skimage.feature
import skimage.filters
from PIL import Image


def array_to_image(x):
    return Image.fromarray(np.uint8(x))


def image_to_array(x):
    return np.array(x, dtype=np.float)


def LBP(img):
    img = array_to_image(img).convert('L')
    lbp = skimage.feature.local_binary_pattern(img, 8, 4)
    return image_to_array(lbp)


def read_image(imgpath):
    img = Image.open(imgpath)
    img_grey = img.convert('L')
    return img, img_grey


# img, img_grey = read_image('_data/fd_001_1.jpg')
# img_grey_np = image_to_array(img_grey)
# x = LBP(img_grey_np)
# array_to_image(x).show()
# print(x)
# print(x.shape)
