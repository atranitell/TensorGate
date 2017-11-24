# -*- coding: utf-8 -*-
""" Assemble all hyper parameters of model
    Author: Kai JIN
    Updated: 2017-11-23
"""


class Net():

  def __init__(self):
    pass

  def cifarnet(self, weight_decay=None, dropout_keep=None):
    """ Cifarnet: input 28x28 images
    """
    self.name = 'cifarnet'
    self.weight_decay = weight_decay
    self.dropout_keep = dropout_keep

  def cgan(self, z_dim=100):
    """ condition gan
    """
    self.z_dim = 100


class Log():

  def __init__(self, print_invl=20,
               save_summaries_invl=2,
               save_model_invl=100,
               test_invl=100,
               val_invl=100,
               max_iter=999999):
    """
    """
    self.print_invl = print_invl
    self.save_summaries_invl = save_summaries_invl
    self.save_model_invl = save_model_invl
    self.test_invl = test_invl
    self.val_invl = val_invl
    self.max_iter = max_iter


class LearningRate():

  def __init__(self):
    pass

  @staticmethod
  def _decay_steps(total_num, batchsize, decay_epochs):
    return int(total_num / batchsize * decay_epochs)

  def fixed(self, learning_rate=0.1):
    """ a constant learning rate type
    """
    self.name = 'fixed'
    self.learning_rate = learning_rate

  def exponential(self, total_num, batchsize,
                  learning_rate=0.1,
                  decay_epochs=30,
                  decay_rate=0.1):
    """ decayed_learning_rate = learning_rate *
          decay_rate ^ (global_step / decay_steps)
    """
    self.name = 'exponential'
    self.learning_rate = learning_rate
    self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)
    self.decay_rate = decay_rate

  def polynomial(self, total_num, batchsize,
                 learning_rate=0.1,
                 decay_epochs=30,
                 end_learning_rate=0.00001):
    self.name = 'polynomial'
    self.learning_rate = learning_rate
    self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)
    self.end_learning_rate = end_learning_rate

  def vstep(self, values=[0.1, 0.01, 0.001, 0.0001],
            boundaries=[10000, 50000, 100000]):
    """ boundaries: [iter1, iter2]
        values: [lr1, lr2, lr3]
    """
    self.name = 'vstep'
    self.learning_rate = values[0]
    self.boundaries = boundaries
    self.values = values

  def natural_exp(self, total_num, batchsize,
                  learning_rate=0.1,
                  decay_rate=0.1,
                  decay_epochs=30):
    self.name = 'natural_exp'
    self.learning_rate = learning_rate
    self.decay_rate = decay_rate
    self.decay_steps = self._decay_steps(total_num, batchsize, decay_epochs)


class Optimizer():

  def __init__(self):
    pass

  def adam(self, beta1=0.9,
           beta2=0.999,
           epsilon=1e-8):
    self.name = 'adam'
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

  def sgd(self):
    self.name = 'sgd'

  def adagrad(self, initial_accumulator_value=0.1):
    self.name = 'adagrad'
    self.initial_accumulator_value = initial_accumulator_value

  def adadelta(self, epsilon=1e-8,
               rho=0.95):
    self.name = 'adadelta'
    self.epsilon = epsilon
    self.rho = rho

  def ftrl(self, learning_rate_power=-0.5,
           initial_accumulator_value=0.1,
           l1=0.0, l2=0.0):
    self.name = 'ftrl'
    self.learning_rate_power = learning_rate_power
    self.initial_accumulator_value = initial_accumulator_value
    self.l1 = l1
    self.l2 = l2

  def momentum(self, momentum=0.9):
    self.name = 'momentum'
    self.momentum = momentum

  def rmsprop(self, decay=0.9,
              momentum=0.0,
              epsilon=1e-10):
    self.name = 'rmsprop'
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon


class Data():

  def __init__(self, batchsize,
               entry_path=None,
               shuffle=None,
               total_num=None,
               loader=None,
               reader_thread=8,
               min_queue_num=128):
    self.batchsize = batchsize
    self.entry_path = entry_path
    self.shuffle = shuffle
    self.total_num = total_num
    self.loader = loader
    self.reader_thread = reader_thread
    self.min_queue_num = min_queue_num
    # for each data attri to allocate a config
    self.configs = []

  def add_image(self, image_config):
    """ add a data attribution to configs  """
    if type(image_config) is type(Image()):
      self.configs.append(image_config)

  def label(self,
            num_classes=10,
            span=10,
            one_hot=False,
            scale=False):
    """ 
    num_classes: net output dim, for regression, the value is 1
    span(range): label span: (0, range)
    one hot: if the input label as a one hot
    scale: True for (0, num_classes) -> (0, 1)
    """
    self.num_classes = num_classes
    self.one_hot = one_hot
    self.range = span
    self.scale = scale


class Image():

  def __init__(self, channels=3,
               frames=1,
               raw_height=None,
               raw_width=None,
               output_height=None,
               output_width=None,
               preprocessing_method=None):
    self.name = 'image'
    self.channels = channels
    self.frames = frames
    self.raw_height = raw_height
    self.raw_width = raw_width
    self.output_height = output_height
    self.output_width = output_width
    self.preprocessing_method = preprocessing_method
