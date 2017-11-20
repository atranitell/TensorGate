# -*- coding: utf-8 -*-
""" SNAPSHOT
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from core.utils.logger import logger


class Snapshot():

  def __init__(self, config, output_dir, name='model'):
    """ config: config['log']
    """
    self.chkp_hook = None
    self.output_dir = output_dir
    self.config = config
    self.hook = None
    self.name = name

  def init(self):
    """ should be init after updater
    """
    self.hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=self.output_dir,
        save_steps=self.config['save_model_iter'],
        saver=tf.train.Saver(var_list=tf.global_variables(),
                             max_to_keep=10000, name='save_all'),
        checkpoint_basename=self.name + '.ckpt')
    return self.hook

  def restore(self, sess, saver):
    ckpt = tf.train.get_checkpoint_state(self.output_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      logger.sys('Load checkpoint from: %s' % ckpt.model_checkpoint_path)
      return global_step
    else:
      return
