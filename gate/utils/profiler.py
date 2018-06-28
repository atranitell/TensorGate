# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/11/21

--------------------------------------------------------

Probe system performance

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

  @staticmethod
  def parameters():
    """ parameters
    """
    param_stats = profiler.profile(
        graph=tf.get_default_graph(),
        cmd='scope',
        options=profiler.ProfileOptionBuilder.trainable_variables_parameter())
    sys.stdout.write('total params: %d\n' % param_stats.total_parameters)

  @staticmethod
  def flops(graph=None):
    """Flops"""
    if graph is None:
      graph = tf.get_default_graph()
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flop = tf.profiler.profile(graph=graph, cmd='op', options=opts)
    if flop is not None:
      print('Total FLOPs: ', flop.total_float_ops)

  @staticmethod
  def time_memory(path, sess, train_op):
    """ time_memory
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
