""" default config class
Resposible for writting config from file
"""
import json
import os


class DatasetBase():

  def __init__(self, extra):
    self.phase = None
    self.train = None
    self.test = None
    self.val = None
    self.config_value = self._load_config_file(extra)

  def set_phase(self, phase):
    """ for switch phase
    """
    if phase == 'train':
      self._train()
    elif phase == 'test':
      self._test()
    elif phase == 'val':
      self._val()
    else:
      raise ValueError('Unknown phase [%s]' % phase)

  def _train(self):
    self.phase = self.train.name
    self.data = self.train.data
    self.lr = self.train.lr
    self.optimizer = self.train.optimizer

  def _test(self):
    self.phase = self.test.name
    self.data = self.test.data

  def _val(self):
    self.phase = self.val.name
    self.data = self.val.data

  def _load_config_file(self, extra):
    if extra is not None:
      with open(extra) as fp:
        return json.load(fp)
    else:
      return None

  def _read_config_file(self, default_v, key_v):
    """ key_v like 'train.lr'
    """
    r = key_v.split('.')
    try:
      if len(r) == 1:
        config_v = self.config_value[r[0]]
      elif len(r) == 2:
        config_v = self.config_value[r[0]][r[1]]
      elif len(r) == 3:
        config_v = self.config_value[r[0]][r[1]][r[2]]
      elif len(r) == 4:
        config_v = self.config_value[r[0]][r[1]][r[2]][r[3]]
      else:
        raise ValueError('Too long to implement!')
      v = config_v
    except:
      v = default_v
    return v
