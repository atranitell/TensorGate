""" A numpy-helper tool to help read and write the npy file.
  It does not a singleton class.
"""
import os
import numpy as np
from core.utils.logger import logger


class NumpyHelper():

  def __init__(self, root_path):
    if not os.path.exists(root_path):
      os.makedirs(root_path)
    self.root_path = root_path
    self.data = {}

  def stack(self, name, arr):
    """ stack on first dims of arr.
    """
    # if type(arr) is not np.ndarray:
    #   arr = np.array(arr)
    if name not in self.data:
      self.data[name] = arr
    else:
      if len(arr.shape) == 1:
        self.data[name] = np.append(self.data[name], arr)
      else:
        self.data[name] = np.row_stack((self.data[name], arr))

  def shape(self, name):
    return self.data[name].shape

  def multi_stack(self, name_list, arr_list):
    for tup in zip(name_list, arr_list):
      self.stack(tup[0], tup[1])

  def dump(self):
    """ dump to file
    """
    for entry in self.data:
      abspath = os.path.join(self.root_path, entry)
      np.save(abspath, self.data[entry])
      logger.info('ndArray content %s has been saved in %s' % (entry, abspath))
