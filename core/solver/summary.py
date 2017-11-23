# -*- coding: utf-8 -*-
""" Summary helper for store items
    Author: Kai JIN
    Updated: 2017-11-20
"""
import tensorflow as tf
from core.utils.logger import logger


class Summary():

  def __init__(self, config):
    """ summary helper
    """
    # setting path
    self.config = config
    self.hook = None
    self.summary = tf.summary.FileWriter(self.config.output_dir)

  def init(self):
    self.hook = tf.train.SummarySaverHook(
        save_steps=self.config.log.save_summaries_invl,
        output_dir=self.config.output_dir,
        summary_op=tf.summary.merge_all())
    return self.hook

  def add(self, tag, value):
    """ add a simple value to summary
    """
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    self.summary.add_summary(summary)

  def adds(self, global_step, tags, values):
    """ add a series of values
    """
    summary = tf.Summary()
    for i, v in enumerate(tags):
      summary.value.add(tag=tags[i], simple_value=values[i])
    self.summary.add_summary(summary, global_step=global_step)
