# -*- coding: utf-8 -*-
""" SNAPSHOT
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from core.utils.logger import logger


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
        saver=tf.train.Saver(var_list=tf.global_variables(),
                             max_to_keep=10000,
                             name='save_all'),
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
      raise ValueError('Could not find suitable restore files.')
