# -*- coding: utf-8 -*-
""" Save Image  - updated: 2017/05/09
"""

import scipy.misc
import numpy as np


def save_images(images, size, image_path):
  """Save mutiple images into one single image.
  Parameters
  -----------
  images : numpy array [batch, w, h, c]
  size : list of two int, row and column number.
      number of images should be equal or less than size[0] * size[1]
  image_path : string.
  Examples
  ---------
  >>> images = np.random.rand(64, 100, 100, 3)
  >>> save_images(images, [8, 8], 'temp.png')
  """
  def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h: j * h + h, i * w:i * w + w, :] = image
    return img

  def imsave(images, size, path):
    # images = (images + 1.) / 2.
    return scipy.misc.imsave(path, merge(images, size))

  return imsave(images, size, image_path)
