# -*- coding: utf-8 -*-
""" Updater
    Author: Kai JIN
    Updated: 2017-08-28
"""
import tensorflow as tf

from core.utils.logger import logger
from core.solver import optimizer
from core.solver import learning_rate


class Updater():

  def __init__(self):
    self.learning_rate = None
    self.optimizer = None
    self.grads = None
    self.saver = None
    self.train_op = None
    self.global_step = None
    self.variables_to_train = None
    self.variables_to_restore = None

  def get_learning_rate(self, cfg_lr=None):
    if self.learning_rate is not None:
      return self.learning_rate
    return learning_rate.configure(cfg_lr, self.global_step)

  def get_optimizer(self, cfg_opt=None):
    if self.optimizer is not None:
      return self.optimizer
    return optimizer.configure(cfg_opt, self.learning_rate)

  def get_train_op(self):
    if self.train_op is not None:
      return self.train_op
    else:
      raise ValueError('train op does not exist.')

  def get_gradients(self, losses=None, cfg_opt=None):
    if self.grads is not None:
      return self.grads
    # compute gradients
    self.grads = self.optimizer.compute_gradients(
        losses, var_list=self.variables_to_train)
    # NOTE: for future to seperate it.
    if cfg_opt.clip_method is 'clip_by_value':
      cmin = cfg_opt.clip_value_min
      cmax = cfg_opt.clip_value_max
      self.grads = [(tf.clip_by_value(grad, cmin, cmax), var)
                    for grad, var in self.grads]
    return self.grads

  def get_global_step(self):
    if self.global_step is not None:
      return self.global_step
    return framework.create_global_step()

  def _inclusion_var(self, exclusions, var_list):
    """ exclude prefix elements of exclusions in var_list
    """
    _vars = []
    for var in var_list:
      excluded = False
      for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
          excluded = True
          break
      if not excluded:
        _vars.append(var)
    return _vars

  def get_trainable_list(self, exclusions=None):
    """ var for train.
    """
    if self.variables_to_train is not None:
      return self.variables_to_train
    if exclusions is not None:
      return self._inclusion_var(exclusions, tf.trainable_variables())
    else:
      return tf.trainable_variables()

  def get_restore_list(self, exclusions=None):
    """ import variables excluded from exclusions.
    """
    if self.variables_to_restore is not None:
      return self.variables_to_restore
    if exclusions is not None:
      return self._inclusion_var(exclusions, tf.global_variables())
    else:
      return tf.global_variables()

  def get_variables_saver(self):
    """ restore list.
    """
    if self.saver is not None:
      return self.saver
    return tf.train.Saver(var_list=self.variables_to_restore,
                          name='restore', allow_empty=True)

  def init_default_updater(self, cfg_lr, cfg_opt, loss, var_list=None):
    """ initialize default updater
    """
    # NOTE need to processing var list

    #
    self.global_step = self.get_global_step()
    self.learning_rate = self.get_learning_rate(cfg_lr)
    self.optimizer = self.get_optimizer(cfg_opt)

    # variable to train
    self.variables_to_train = self.get_trainable_list()

    # compute gradients
    self.grads = self.get_gradients(loss, cfg_opt)
    grad_op = self.optimizer.apply_gradients(
        self.grads, self.global_step, name='train_step')

    self.train_op = grad_op

    self.variables_to_restore = self.get_restore_list()
    self.saver = self.get_variables_saver()
