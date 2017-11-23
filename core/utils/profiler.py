# -*- coding: utf-8 -*-
""" Probe system performance
    Author: Kai JIN
    Update: 17/11/22
"""

import sys
import tensorflow as tf
from tensorflow import profiler


class Profiler():
  """ The class to manage the probe tools, offering:
  1) network structure (parameter number)
  2) FLOPs
  3) time and memory analyzeing
  """

  def __init__(self):
    pass

  @staticmethod
  def parameters():
    """
    """
    param_stats = profiler.profile(
        graph=tf.get_default_graph(),
        cmd='scope',
        options=profiler.ProfileOptionBuilder.trainable_variables_parameter())
    sys.stdout.write('total params: %d\n' % param_stats.total_parameters)

  @staticmethod
  def flops():
    """
    """
    param_stats = tf.profiler.profile(
        graph=tf.get_default_graph(),
        cmd='scope',
        options=tf.profiler.ProfileOptionBuilder.float_operation())
    sys.stdout.write('total flops: %d\n' % param_stats.total_float_ops)

  @staticmethod
  def time_memory(path, sess, train_op):
    """
    """
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()
    with tf.contrib.tfprof.ProfileContext(path,
                                          trace_steps=range(10, 20),
                                          dump_steps=[20]) as pctx:
      pctx.trace_next_step()
      pctx.dump_next_step()
      sess.run(train_op)
      pctx.profiler.profile_operations(options=opts)
