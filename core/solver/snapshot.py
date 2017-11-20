# -*- coding: utf-8 -*-
""" SNAPSHOT
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf
from core.utils.logger import logger


class Snapshot():

  def __init__(self, config):
    self.summary = None
    self.summary_hook = None
    self.chkp_hook = None
    self.config = config

  def get_chkp_hook(self, name, chkp):
    if self.chkp_hook is None:
      self.chkp_hook = tf.train.CheckpointSaverHook(
          checkpoint_dir=chkp,
          save_steps=self.config['save_model_iter'],
          saver=tf.train.Saver(var_list=tf.global_variables(),
                               max_to_keep=10000, name='save_all'),
          checkpoint_basename=name + '.ckpt')
    return self.chkp_hook

  def get_summary_hook(self, chkp):
    if self.summary_hook is None:
      self.summary_hook = tf.train.SummarySaverHook(
          save_steps=self.config['save_summaries_iter'],
          output_dir=chkp,
          summary_op=tf.summary.merge_all())
    return self.summary_hook

  def get_summary(self, chkp):
    if self.summary is None:
      self.summary = tf.summary.FileWriter(chkp)
    return self.summary

  def restore(self, sess, chkp, saver):
    ckpt = tf.train.get_checkpoint_state(chkp)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      logger.sys('Load checkpoint from: %s' % ckpt.model_checkpoint_path)
      return global_step
    else:
      return
