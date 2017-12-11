# -*- coding: utf-8 -*-
""" Database class for institude of classic pipline
  Author: Kai JIN
  Updated: 2017-12-11
"""
import time
import random
from core.utils.logger import logger
from core.database import data_entry
from PIL import Image
import numpy as np
import tensorflow as tf


class DataBase(object):

  def __init__(self, config):
    """ empty
    """
    self._cur = 0
    self._size = 0
    self._name = config.name
    self._phase = config.phase
    self._entry_path = config.data.entry_path
    self._shuffle = config.data.shuffle
    self._batchsize = config.data.batchsize
    self._index = []
    self.load_contents()

    self.use_active_sampler = True

  def load_contents(self):
    """ parse all data entry and load in system
      and store in numpy-form big array
    """
    _start_time = time.time()
    res, count = data_entry.parse_from_text(
        self._entry_path, (str, float), (True, False))
    self._size = count

    self.r_data = []
    self.r_label = []
    self.r_path = []
    per_probs = 1.0 / self._size

    for idx in range(self._size):
      with Image.open(res[0][idx]) as source:
        self.r_data.append(np.array(source))
      self.r_label.append(res[1][idx])
      self.r_path.append(res[0][idx])
      self._index.append([idx, per_probs])

    self.r_data = np.array(self.r_data)
    self.r_label = np.array(self.r_label)
    self.r_path = np.array(self.r_path)

    _dur_time = time.time() - _start_time
    logger.info('Loading data %s, label %s, elapsed %d s.' %
                (str(self.r_data.shape), str(self.r_label.shape),
                 _dur_time))

  def loads(self):
    # return placeholder tensor
    data = tf.placeholder(tf.float32, (None,) + self.r_data.shape[1:])
    label = tf.placeholder(tf.int32, (None,))
    path = tf.placeholder(tf.string, (None,))
    return data, label, path

  def reset_index(self):
    """ reset database index by ascend order
    """
    self._index.sort()
    self._cur = 0

  def is_reach_to_end(self):
    """ before prefetch next batch, the system will detect if the index has 
      reached the end.
    """
    return True if self._cur + self._batchsize > self._size else False

  def shuffle_index(self):
    """
    """
    random.shuffle(self._index)

  def prob_pick(self):
    batch_index = []
    s = sum([i[1] for i in self._index])
    for _ in range(self._batchsize):
      c = 0.0
      x = random.uniform(0, s)
      for item, p in self._index:
        c += p
        if x <= c:
          batch_index.append(item)
          break
    return batch_index

  def pick(self):
    batch_index = []
    for i in self._index[self._cur:self._cur + self._batchsize]:
      batch_index.append(i[0])
    return batch_index

  def gen_prob(self, loss_batch):
    scale = sum([self._index[index][1] for index in self.batch_index])
    for i, index in enumerate(self.batch_index):
      self._index[index][1] = scale * loss_batch[i]
    # print(scale, sum([self._index[i][1] for i in self.batch_index]))

  def next_batch(self, data, label, path):
    """
    """
    # check limits
    if self.is_reach_to_end():
      self._cur = 0
      if self._shuffle:
        self.shuffle_index()

    if self._phase == 'train' and self.use_active_sampler:
      self.batch_index = self.prob_pick()
    else:
      self.batch_index = self.pick()
    batch_data = [self.r_data[i] for i in self.batch_index]
    batch_label = [self.r_label[i] for i in self.batch_index]
    batch_path = [self.r_path[i] for i in self.batch_index]

    self._cur += self._batchsize
    return {data: batch_data, label: batch_label, path: batch_path}
