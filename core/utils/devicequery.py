""" It will collect a set of device performance tools.
  Update: 2017-12-09
  Author: Kai JIN
"""
from tensorflow.python.client import device_lib as _device_lib


def showing_avaliable_device():
  for x in _device_lib.list_local_devices():
    print(x)
