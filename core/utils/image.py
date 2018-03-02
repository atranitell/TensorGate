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

  # def heatmap(self, path, data, weight, raw_h, raw_w, save_path=''):
  #   """ a uniform IO for generating a single heatmap
  #   Input
  #       ImagePath: a str of path to load image
  #       data: (C, kernelsize, kernelsize)
  #       weights: (C, )
  #       raw_h: height of Image
  #       raw_w: width of Image
  #   """
  #   # dim
  #   channels = data.shape[0]

  #   conv_img = data[0] * weight[0]
  #   for _c in range(1, channels):
  #     conv_img += data[_c] * weight[_c]
  #   conv_img = cv2.resize(conv_img, (raw_h, raw_w))

  #   # get raw image
  #   # src = cv2.imread(path)
  #   # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

  #   # heatmap-and-image
  #   # plt.close('all')
  #   # plt.xticks([], [])
  #   # plt.yticks([], [])
  #   # plt.imshow(src)
  #   # im = plt.imshow(conv_img, cmap='jet', alpha=0.4, interpolation='nearest')
  #   # plt.savefig(save_path + '.jpg', bbox_inches='tight')

  #   # heatmap-only
  #   plt.close('all')
  #   plt.xticks([], [])
  #   plt.yticks([], [])
  #   im = plt.imshow(conv_img, cmap='jet', interpolation='nearest')
  #   plt.colorbar(im)
  #   plt.savefig(save_path + '_heatmap.jpg', bbox_inches='tight')


image = Image()
