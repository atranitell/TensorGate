# -*- coding: utf-8 -*-
""" Save Image  - updated: 2017/05/09
"""
import scipy.misc
import numpy as np
import math
from core.utils.path import path
# import cv2
# from matplotlib import pyplot as plt


class Image():
  """ package image operation
  """

  def save(self, _img, _path):
    """ Save a numpy array to path.
      Need not scale to 255.
    """
    return scipy.misc.imsave(_path, _img)

  def save_multi(self, _dst, _imgs, _paths):
    """ save all imgs to dst dir with name(paths)
    """
    for idx, content in enumerate(_imgs):
      img_path = path.join_name(_dst, _paths[idx])
      self.save(content, img_path)

  def save_multi_to_one(self, _imgs, _size, _path):
    """Save mutiple images into one single image.
    Parameters
    -----------
    images : numpy array [batch, w, h, c]
    size : list of two int, row and column number.
        number of images should be equal or less than size[0] * size[1]
    path : string.
    Examples
    ---------
    >>> images = np.random.rand(64, 100, 100, 3)
    >>> save_images(images, [8, 8], 'temp.png')
    """
    def merge_images(images, size):
      h, w = images.shape[1], images.shape[2]
      img = np.zeros((h * size[0], w * size[1], 3))
      for idx, _img in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h: j * h + h, i * w:i * w + w, :] = _img
      return img
    return self.save(merge_images(_imgs, _size), _path)

  @staticmethod
  def max_factorization(num):
    """ factorize a num to make the maximum sum
      return: factor1 <= factor2
    """
    max_factor_1 = 1
    for i in range(1, int(math.sqrt(num)) + 1):
      if num % i == 0:
        max_factor_1 = i
    max_factor_2 = int(num / max_factor_1)
    return max_factor_1, max_factor_2

  def save_batch_to_one(self, imgs, batchsize, dst, name):
    """ save a batch to one images
    """
    size_l, size_w = self.max_factorization(batchsize)
    abspath = path.join(dst, name, 'png')
    self.save_multi_to_one(imgs, [size_l, size_w], abspath)

  def save_batchs_to_one(self, imgs_list, batchsize, dst, step, name_list):
    """ save batch to one image for multi name-list
      imgs_list: [img_batch1, img_batch2, ...]
      name_list: ['img1', 'img2', ...]
    """
    size_l, size_w = self.max_factorization(batchsize)
    for i, imgs in enumerate(imgs_list):
      abspath = path.join_step(dst, step, 'png', name_list[i])
      self.save_multi_to_one(imgs, [size_l, size_w], abspath)


image = Image()
