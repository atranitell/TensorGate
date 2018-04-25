# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/25

--------------------------------------------------------

PARAMS COLLECTIONS:
  Package a series of parameter to a collections, which is conveniently to
  standard the config file.

- Net:
- Log:
- Opt:
- Lr:
- Data:

"""

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers


class NET():
  """ we have to consider a situation - the users maybe use multi-combination
    of networks. therefore, we should define a list to store all net config.

  self.net = [NET().vgg11(...)]

  """

  def __init__(self):
    self.name = None

  def _set_dropout_keep(self, dropout_keep=0.5):
    self.dropout_keep = dropout_keep

  def _set_weight_decay(self, weight_decay=0.0001):
    self.weight_decay = weight_decay

  def _set_dropout_keep(self, dropout_keep=0.5):
    self.dropout_keep = dropout_keep

  def _set_batch_norm(self,
                      batch_norm_decay=0.997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True,
                      use_batch_norm=True):
    """ USE Batch Normalization """
    self.use_batch_norm = use_batch_norm
    self.batch_norm_decay = batch_norm_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.batch_norm_scale = batch_norm_scale

  def _set_activation_fn(self, name='relu'):
    """ choose different activation function """
    if name is 'relu':
      self.activation_fn = tf.nn.relu
    elif name is 'sigmoid':
      self.activation_fn = tf.nn.sigmoid
    elif name is 'tanh':
      self.activation_fn = tf.nn.tanh
    elif name is 'elu':
      self.activation_fn = tf.nn.elu
    elif name is 'tanh':
      self.activation_fn = tf.nn.tanh
    else:
      raise ValueError('Unknown activation fn [%s]' % name)

  def _set_initializer_fn(self, name='xavier'):
    """ choose differnet initializer for some certain task
    """
    if name is 'zeros':
      self.initializer_fn = tf.zeros_initializer
    elif name is 'orthogonal':
      self.initializer_fn = tf.orthogonal_initializer
    elif name is 'normal':
      self.initializer_fn = tf.truncated_normal_initializer
    elif name is 'xavier':
      self.initializer_fn = layers.xavier_initializer
    elif name is 'uniform':
      self.initializer_fn = tf.random_uniform_initializer
    else:
      raise ValueError('Unknown input type %s' % name)

  def _set_units_and_layers(self, n_units=[128], n_layers=1):
    """ choose the rnn layers and units
    """
    self.num_layers = n_layers
    self.num_units = n_units

  def _set_cell_fn(self, name='gru'):
    """ choose different rnn cell
    """
    if name is 'rnn':
      self.cell_fn = rnn.BasicRNNCell
    elif name is 'gru':
      self.cell_fn = rnn.GRUCell
    elif name is 'basic_lstm':
      self.cell_fn = rnn.BasicLSTMCell
    elif name is 'lstm':
      self.cell_fn = rnn.LSTMCell
    else:
      raise ValueError('Unknown input type %s' % name)

  def _set_global_pool(self, global_pool=True):
    self.global_pool = global_pool

  def _raise_if_net_defined(self):
    if self.name is not None:
      raise ValueError('The network has been defined.')

  def _set_name(self, name):
    self._raise_if_net_defined()
    self.name = name

  def _set_num_classes(self, num_classes):
    """ determined the output dim
    """
    self.num_classes = num_classes

  def resnet_v2_50(self,
                   num_classes=10,
                   weight_decay=0.0001,
                   batch_norm_decay=0.997,
                   batch_norm_epsilon=1e-5,
                   batch_norm_scale=True,
                   use_batch_norm=True,
                   activation_fn='relu',
                   global_pool=True,
                   scope='resnet_v2_50'):
    self._set_name(scope)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_batch_norm(batch_norm_decay,
                         batch_norm_epsilon,
                         batch_norm_scale,
                         use_batch_norm)
    self._set_activation_fn(activation_fn)
    self._set_global_pool(global_pool)

  def cifarnet(self,
               num_classes=10,
               weight_decay=0.004,
               dropout_keep=0.5):
    self._set_name('cifarnet')
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)

  def lenet(self,
            num_classes=10,
            weight_decay=0.004,
            dropout_keep=0.5):
    self._set_name('lenet')
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)


class LOG():

  def __init__(self,
               print_invl=20,
               save_summary_invl=20,
               save_model_invl=1000,
               test_invl=1000,
               val_invl=1000,
               max_iter=1000000):
    self.print_invl = print_invl
    self.save_summary_invl = save_summary_invl
    self.save_model_invl = save_model_invl
    self.test_invl = test_invl
    self.val_invl = val_invl
    self.max_iter = max_iter


class OPT():

  def set_adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.name = 'adam'
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def set_sgd(self):
    self.name = 'sgd'

  def set_adagrad(self, initial_accumulator_value=0.1):
    self.name = 'adagrad'
    self.initial_accumulator_value = initial_accumulator_value

  def set_adadelta(self, epsilon=1e-8, rho=0.95):
    self.name = 'adadelta'
    self.epsilon = epsilon
    self.rho = rho

  def set_ftrl(self,
               learning_rate_power=-0.5,
               initial_accumulator_value=0.1,
               l1=0.0, l2=0.0):
    self.name = 'ftrl'
    self.learning_rate_power = learning_rate_power
    self.initial_accumulator_value = initial_accumulator_value
    self.l1 = l1
    self.l2 = l2

  def set_momentum(self, momentum=0.9):
    self.name = 'momentum'
    self.momentum = momentum

  def set_rmsprop(self, decay=0.9, momentum=0.0, epsilon=1e-10):
    self.name = 'rmsprop'
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon


class LR():
  """ some leraning rate policy may use the total number.
  """

  # @staticmethod
  # def _decay_steps(total_num, batchsize, decay_epochs):
  #   return int(total_num / batchsize * decay_epochs)

  def set_fixed(self, learning_rate=0.1):
    """ a constant learning rate type
    """
    self.name = 'fixed'
    self.learning_rate = learning_rate

  def set_exponential(self,
                      learning_rate=0.1,
                      decay_epochs=30,
                      decay_rate=0.1):
    """ decayed_learning_rate = learning_rate *
          decay_rate ^ (global_step / decay_steps)
    """
    self.name = 'exponential'
    self.learning_rate = learning_rate
    self.decay_epochs = decay_epochs
    # self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)
    self.decay_rate = decay_rate

  def set_polynomial(self,
                     learning_rate=0.1,
                     decay_epochs=30,
                     end_learning_rate=0.00001):
    self.name = 'polynomial'
    self.learning_rate = learning_rate
    self.decay_epochs = decay_epochs
    # self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)
    self.end_learning_rate = end_learning_rate

  def set_vstep(self,
                values=[0.1, 0.01, 0.001, 0.0001],
                boundaries=[10000, 50000, 100000]):
    """ boundaries: [iter1, iter2]
        values: [lr1, lr2, lr3]
    """
    self.name = 'vstep'
    self.learning_rate = values[0]
    self.boundaries = boundaries
    self.values = values

  def set_natural_exp(self,
                      learning_rate=0.1,
                      decay_rate=0.1,
                      decay_epochs=30):
    self.name = 'natural_exp'
    self.learning_rate = learning_rate
    self.decay_rate = decay_rate
    self.decay_epochs = decay_epochs
    # self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)


class DATA():

  def __init__(self,
               batchsize,
               entry_path=None,
               shuffle=None,
               name='Data'):
    self.name = name
    self.batchsize = batchsize
    self.entry_path = entry_path
    self.shuffle = shuffle
    self.total_num = None
    self.configs = []

  def set_queue_loader(self,
                       loader=None,
                       reader_thread=8,
                       min_queue_num=128):
    self.loader = loader
    self.reader_thread = reader_thread
    self.min_queue_num = min_queue_num

  def set_entry_attr(self,
                     entry_dtype=None,
                     entry_check=None):
    """ entry_dtype: a tuple showing the item type
      entry_check: a tuple describes a file path
    """
    self.entry_dtype = entry_dtype
    self.entry_check = entry_check

  def set_label(self,
                num_classes=None,
                ranges=None,
                one_hot=False,
                scale=False):
    """ num_classes: net output dim, for regression, the value is 1
      ranges: label span: [0, ranges]
      one hot: if the input label as a one hot
      scale: True for ranges -> (0, 1) and ranges is not None
    """
    self.num_classes = num_classes
    self.one_hot = one_hot
    self.range = ranges
    self.scale = scale

  def add(self, config):
    self.configs.append(config)


class Image():

  def set_fixed_length_image(self,
                             channels=3,
                             frames=1,
                             raw_height=None,
                             raw_width=None,
                             output_height=None,
                             output_width=None,
                             preprocessing_method=None,
                             gray=False,
                             name='fixed_length_image'):
    self.name = name
    self.channels = channels
    self.frames = frames
    self.raw_height = raw_height
    self.raw_width = raw_width
    self.output_height = output_height
    self.output_width = output_width
    self.preprocessing_method = preprocessing_method
    self.gray = gray


class Numpy():

  def __init__(self,
               shape=None,
               name='numpy_type_data'):
    self.name = name
    self.shape = shape


class Phase():

  def __init__(self, name):
    self.name = name