# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PARAMS COLLECTIONS: Package a series of parameter to a collections, 
  which is conveniently to standard the config file.
"""

import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers


class NET():

  def __init__(self):
    self.name = None

  def _set_spatial_squeeze(self, spatial_squeeze=True):
    self.spatial_squeeze = spatial_squeeze

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
                      use_batch_norm=True,
                      use_pre_batch_norm=False):
    """USE Batch Normalization"""
    self.use_batch_norm = use_batch_norm
    self.use_pre_batch_norm = use_pre_batch_norm
    self.batch_norm_decay = batch_norm_decay
    self.batch_norm_epsilon = batch_norm_epsilon
    self.batch_norm_scale = batch_norm_scale

  def _set_activation_fn(self, name='relu'):
    """Choose different activation function"""
    if name == 'relu':
      self.activation_fn = tf.nn.relu
    elif name == 'sigmoid':
      self.activation_fn = tf.nn.sigmoid
    elif name == 'tanh':
      self.activation_fn = tf.nn.tanh
    elif name == 'elu':
      self.activation_fn = tf.nn.elu
    elif name == 'tanh':
      self.activation_fn = tf.nn.tanh
    elif name == 'leaky_relu':
      self.activation_fn = tf.nn.leaky_relu
    else:
      raise ValueError('Unknown activation fn [%s]' % name)

  def _set_initializer_fn(self, name='xavier'):
    """Choose differnet initializer for some certain task"""
    if name == 'zeros':
      self.initializer_fn = tf.zeros_initializer
    elif name == 'orthogonal':
      self.initializer_fn = tf.orthogonal_initializer
    elif name == 'normal':
      self.initializer_fn = tf.truncated_normal_initializer
    elif name == 'xavier':
      self.initializer_fn = layers.xavier_initializer
    elif name == 'uniform':
      self.initializer_fn = tf.random_uniform_initializer
    else:
      raise ValueError('Unknown input type %s' % name)

  def _set_units_and_layers(self, n_units=[128], n_layers=1):
    """Choose the rnn layers and units"""
    self.num_layers = n_layers
    self.num_units = n_units

  def _set_cell_fn(self, name='gru'):
    """Choose different rnn cell"""
    if name == 'rnn':
      self.cell_fn = rnn.BasicRNNCell
    elif name == 'gru':
      self.cell_fn = rnn.GRUCell
    elif name == 'basic_lstm':
      self.cell_fn = rnn.BasicLSTMCell
    elif name == 'lstm':
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
    """Determined the output dim"""
    self.num_classes = num_classes

  def _set_z_dim(self, z_dim):
    self.z_dim = z_dim

  def resnet_v2(self,
                depth='50',  # 50, 101, 152, 200
                num_classes=1000,
                weight_decay=0.0001,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True,
                use_batch_norm=True,
                activation_fn='relu',
                global_pool=True,
                scope='resnet_v2_'):
    self._set_name(scope+depth)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_batch_norm(batch_norm_decay,
                         batch_norm_epsilon,
                         batch_norm_scale,
                         use_batch_norm)
    self._set_activation_fn(activation_fn)
    self._set_global_pool(global_pool)

  def resnet_v2_critical(self,
                depth='50',  # 50, 101, 152, 200
                num_classes=1000,
                weight_decay=0.0001,
                batch_norm_decay=0.997,
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True,
                use_batch_norm=True,
                activation_fn='relu',
                global_pool=True,
                scope='resnet_v2_critical_'):
    self._set_name(scope+depth)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_batch_norm(batch_norm_decay,
                         batch_norm_epsilon,
                         batch_norm_scale,
                         use_batch_norm)
    self._set_activation_fn(activation_fn)
    self._set_global_pool(global_pool)

  def resnet_v2_bishared(self,
                         depth='50',  # 50, 101, 152, 200
                         num_classes=1000,
                         weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True,
                         use_batch_norm=True,
                         activation_fn='relu',
                         global_pool=True,
                         scope='resnet_v2_'):
    self._set_name(scope+depth+'_bishared')
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

  def alexnet(self,
              num_classes=1000,
              weight_decay=0.0005,
              dropout_keep=0.5,
              spatial_squeeze=True,
              global_pool=False,
              scope='alexnet_v2'):
    self._set_name(scope)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)
    self._set_global_pool(global_pool)
    self._set_spatial_squeeze(spatial_squeeze)

  def alexnet_bishared(self,
                       num_classes=1000,
                       weight_decay=0.0005,
                       dropout_keep=0.5,
                       spatial_squeeze=True,
                       global_pool=True,
                       scope='alexnet_v2_bishared'):
    self._set_name(scope)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)
    self._set_global_pool(global_pool)
    self._set_spatial_squeeze(spatial_squeeze)

  def audionet(self,
               num_classes=1,
               weight_decay=0.0001,
               batch_norm_decay=0.997,
               batch_norm_epsilon=1e-5,
               batch_norm_scale=True,
               use_batch_norm=True,
               activation_fn='relu',
               global_pool=True,
               scope='audionet'):
    self._set_name(scope)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_batch_norm(batch_norm_decay,
                         batch_norm_epsilon,
                         batch_norm_scale,
                         use_batch_norm)
    self._set_activation_fn(activation_fn)
    self._set_global_pool(global_pool)

  def sensnet(self,
              num_classes=1,
              weight_decay=0.0001,
              unit_type='multi_addition',
              unit_num=[1, 1, 1, 1],
              batch_norm_decay=0.999,
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              use_batch_norm=False,
              use_pre_batch_norm=False,
              dropout_keep=0.5,
              activation_fn='leaky_relu',
              version='sensnet_v1'):
    self._set_name(version)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_batch_norm(batch_norm_decay,
                         batch_norm_epsilon,
                         batch_norm_scale,
                         use_batch_norm,
                         use_pre_batch_norm)
    self._set_activation_fn(activation_fn)
    self._set_dropout_keep(dropout_keep)
    self.unit_type = unit_type
    self.unit_num = unit_num

  def vgg(self,
          depth='11',  # 11, 16, 19
          num_classes=1000,
          weight_decay=0.0005,
          dropout_keep=0.5,
          spatial_squeeze=True,
          scope='vgg_',
          # fc_conv_padding='VALID',
          global_pool=False):
    self._set_name(scope+depth)
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)
    self._set_spatial_squeeze(spatial_squeeze)
    self._set_global_pool(global_pool)

  def vgg_bishared(self,
                   depth='11',
                   num_classes=1000,
                   weight_decay=0.0005,
                   dropout_keep=0.5,
                   spatial_squeeze=True,
                   scope='vgg_',
                   # fc_conv_padding='VALID',
                   global_pool=False):
    self._set_name(scope+depth+'_bishared')
    self._set_num_classes(num_classes)
    self._set_weight_decay(weight_decay)
    self._set_dropout_keep(dropout_keep)
    self._set_spatial_squeeze(spatial_squeeze)
    self._set_global_pool(global_pool)

  def kinvae(self,
             z_dim=100,
             scope='kinvae'):
    self._set_name(scope)
    self._set_z_dim(z_dim)


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
  """Some leraning rate policy may use the total number."""

  def set_fixed(self, learning_rate=0.1):
    """a constant learning rate type"""
    self.name = 'fixed'
    self.learning_rate = learning_rate

  def set_exponential(self,
                      learning_rate=0.1,
                      decay_epochs=30,
                      decay_rate=0.1):
    self.name = 'exponential'
    self.learning_rate = learning_rate
    self.decay_epochs = decay_epochs
    self.decay_rate = decay_rate

  def set_polynomial(self,
                     learning_rate=0.1,
                     decay_epochs=30,
                     end_learning_rate=0.00001):
    self.name = 'polynomial'
    self.learning_rate = learning_rate
    self.decay_epochs = decay_epochs
    self.end_learning_rate = end_learning_rate

  def set_vstep(self,
                values=[0.1, 0.01, 0.001, 0.0001],
                boundaries=[10000, 50000, 100000]):
    """boundaries: [iter1, iter2] values: [lr1, lr2, lr3]"""
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

  def set_custom_loader(self,
                        loader=None,
                        data_dir=None):
    self.loader = loader
    self.data_dir = data_dir

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
    self.entry_dtype = entry_dtype
    self.entry_check = entry_check

  def set_label(self,
                num_classes=None,
                span=None,
                one_hot=False,
                scale=False):
    """Set label information.
    Args:
      num_classes: net output dim, for regression, the value is 1
      span: label span: [0, span]
      one hot: if the input label as a one hot
      scale: True for span -> (0, 1) and span is not None
    """
    self.num_classes = num_classes
    self.one_hot = one_hot
    self.span = span
    self.scale = scale

  def add(self, config):
    if isinstance(config, list):
      self.configs += config
    else:
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


class Audio():

  def set_fixed_length_audio(self,
                             frame_num=32,
                             frame_length=200,
                             frame_invl=200,
                             name='fixed_length_image'):
    self.name = name
    # load number of frame
    self.frame_num = frame_num
    # the length of each frame
    self.frame_length = frame_length
    # the interval of slide window
    # if the value is equal to frame length, each frame will not overlap
    self.frame_invl = frame_invl


class Numpy():

  def __init__(self,
               shape=None,
               name='numpy_type_data'):
    self.name = name
    self.shape = shape


class Phase():

  def __init__(self, name):
    self.name = name
