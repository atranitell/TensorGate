# -*- coding: utf-8 -*-
""" Save Image  - updated: 2017/05/09
"""
import scipy.misc
import numpy as np
from PIL import Image
from core import utils


def saveall(dst, imgs, paths):
  """ save all imgs to dst dir with name(paths)
  """
  for idx, content in enumerate(imgs):
    dst = utils.path.join_name(dst, paths[idx])
    save(content, dst)


def save(image, path):
  """ Save a numpy array to path.
    Need not scale to 255.
  """
  return scipy.misc.imsave(path, image)


def save_images(images, size, path):
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
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h: j * h + h, i * w:i * w + w, :] = image
    return img
  return save(merge_images(images, size), path)
