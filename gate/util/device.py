# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2017/12/09

--------------------------------------------------------

It will collect a set of device performance tools.

"""

from tensorflow.python.client import device_lib


def showing_avaliable_device():
  """ showing the available device.
  """
  for x in device_lib.list_local_devices():
    print(x)
