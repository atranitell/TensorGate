# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/8/28

--------------------------------------------------------

Snapshot

"""

import tensorflow as tf
from gate.util.logger import logger


class Snapshot():

  def __init__(self, config):
    """ config
    """
    self.config = config
    self.chkp_hook = None
    self.hook = None
    self.name = config.name

  def init(self):
    """ should be init after updater
    """
    self.hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=self.config.output_dir,
        save_steps=self.config.log.save_model_invl,
        saver=tf.train.Saver(
            var_list=tf.global_variables(), max_to_keep=10000),
        checkpoint_basename=self.name + '.ckpt')
    return self.hook

  def restore(self, sess, saver):
    ckpt = tf.train.get_checkpoint_state(self.config.output_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      try:
        int(global_step)
      except:
        global_step = 0
      logger.sys('Load checkpoint from: %s' % ckpt.model_checkpoint_path)
      return global_step
    else:
      logger.sys('Start a new model to train.')
      return
