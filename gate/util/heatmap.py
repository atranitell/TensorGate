# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

HEATMAP
  code from: https://github.com/insikk/Grad-CAM-tensorflow/

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from gate.util import filesystem


class HeatMap():
  """ heatmap assemble
  """

  def __init__(self, raw_height=None, raw_width=None):
    self.raw_height = raw_height
    self.raw_width = raw_width

  def _save(self, hmap, method, dstpath=None, srcpath=None, overlap=False):
    plt.close('all')
    plt.xticks([], [])
    plt.yticks([], [])

    if overlap:
      src = cv2.imread(srcpath)
      src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
      src = cv2.resize(src, (self.raw_height, self.raw_width))
      plt.imshow(src)

    if method == 'jet':
      hmap = cv2.resize(hmap, (self.raw_height, self.raw_width))
      plt.imshow(hmap, cmap='jet', alpha=0.8, interpolation='nearest')

    if method == 'gray':
      plt.imshow(hmap, cmap='gray')

    plt.savefig(dstpath, bbox_inches='tight', pad_inches=-0.1)

  def make_paths(self, dstdir, paths):
    """ 
    paths is bytes-format string:
      [b'../_datasets/ImageNet\\cat.jpg',
       b'../_datasets/ImageNet\\dog.jpg']
    dstdir is heatmap folder: '../_outputs/model/heatmap/

    Return: a list of str:
    dst_list:
      ['../_outputs/model/heatmap/cat.jpg',
       '../_outputs/model/heatmap/dog.jpg']
    src_list:
      ['../_datasets/ImageNet\\cat.jpg',
       '../_datasets/ImageNet\\dog.jpg']
    """
    dst_list = []
    src_list = []
    for _path in paths:
      _path = str(_path, encoding='utf-8')
      src_list.append(_path)
      dst_list.append(filesystem.join_name(dstdir, _path))
    return dst_list, src_list

  def gap(self, conv_data, fc_weight, pos,
          dstpaths=None, srcpaths=None, overlap=False):
    """ GAP method should use a fixed GAP structure:
    Input:
      data->conv layer->global average pooling->softmax
            (n,h,w,c)                          (c,1000)
      conv_data: (n, h, w, c) is a numpy-array via last conv layer processing
      fc_weight: (n, c, 1000/?) is numpy-array of last-fc-layer
      pos: (n,) is choose a classification (general using predication)
      dstpaths: (n,)
      overlap: True for combine source image and heatmap image in one map.

    Return:
      a array-like image content

    Demo:
      loss, error, pred = self._loss(logit, label)
      x = net['resnet_v2_50/block4/unit_3/bottleneck_v2']
      w = variables.select_vars('resnet_v2_50/logits')[0]
      heatmap = HeatMap(224, 224)

      _x, _w, _p, _path = sess.run([x, w, pred, path])
      srcpaths, dstpaths = heatmap.make_paths(test_dir, _path)
      heatmap.gap(_x, _w, _p, dstpaths, srcpaths)
    """
    assert len(conv_data.shape) == 4
    assert len(fc_weight.shape) == 3
    content = []
    for n in range(conv_data.shape[0]):
      gap_img = conv_data[n, :, :, 0] * fc_weight[n, 0, pos[n]]
      for c in range(1, conv_data.shape[3]):
        gap_img += conv_data[n, :, :, c] * fc_weight[n, c, pos[n]]
      content.append(gap_img)

      if dstpaths is not None:
        self._save(gap_img, 'jet', dstpaths[n], srcpaths[n], overlap)

    return np.array(content)

  def grad_cam(self, conv_data, conv_grad,
               dstpaths=None, srcpaths=None, overlap=False):
    """ grad cam
    Demo:
      heatmap = HeatMap(self.data.configs[0].raw_height,
                        self.data.configs[0].raw_width)
      x = net['gap_conv']
      g = tf.gradients(logit, x)[0]
      _x, _g, _path = sess.run([x, g, path])
      dstlist, srclist = heatmap.make_paths(test_dir, _path)
      heatmap.grad_cam(_x, _g, dstlist, srclist, True)
    """
    content = []
    for n in range(conv_data.shape[0]):
      x = conv_data[n]
      g = conv_grad[n]

      weights = np.mean(g, axis=(0, 1))
      cam = np.zeros(x.shape[0: 2], dtype=np.float32)
      for i, w in enumerate(weights):
        cam += w * x[:, :, i]

      cam = np.maximum(cam, 0)
      cam = cam / np.max(cam)

      content.append(cam)
      if dstpaths is not None:
        self._save(cam, 'jet', dstpaths[n], srcpaths[n], overlap)

    return np.array(content)

  def guided_backpropagation(self, gb, dstpaths=None):
    """ the high-resolution gradient image should use the image that
      does not do image augmentation (substract mean or others).
    Input:
      gb: (n, raw_height, raw_width, 3/1) grad of image

    Demo:
      heatmap = HeatMap(224, 224)
      gb_grad = tf.gradients(tf.reduce_max(logit, 1), data)[0]
      _gb, _path = sess.run([gb_grad, path])
      srclist, dstlist= heatmap.make_paths(test_dir, _path)
      heatmap.guided_backpropagation(_gb, dstlist)
    """
    content = []
    for n in range(gb.shape[0]):
      gb_img = gb[n]
      gb_img -= np.min(gb_img)
      gb_img /= gb_img.max()
      content.append(gb_img)
      if dstpaths is not None:
        self._save(gb_img, 'gray', dstpaths[n], None, False)
    return np.array(content)

  def guided_grad_cam(self, conv_data, conv_grad, gb,
                      dstpaths=None, srcpaths=None, overlap=False):
    """
    """
    cams = self.grad_cam(conv_data, conv_grad)
    gb = self.guided_backpropagation(gb)

    content = []
    for n in range(cams.shape[0]):
      cam = cv2.resize(cams[n], (self.raw_height, self.raw_width))
      gd_gb = np.dstack((
          gb[n, :, :, 0] * cam,
          gb[n, :, :, 1] * cam,
          gb[n, :, :, 2] * cam))
      content.append(gd_gb)

      if dstpaths is not None:
        self._save(gd_gb, 'jet', dstpaths[n], srcpaths[n], overlap)

    return gd_gb
