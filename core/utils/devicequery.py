""" It will collect a set of device performance tools.
  Update: 2017-12-09
  Author: Kai JIN
"""
from tensorflow.python.client import device_lib as _device_lib


class DeviceQuery():
  """ output the device info
  """

  def showing_avaliable_device(self):
    """ showing the available device.
    """
    for x in _device_lib.list_local_devices():
      print(x)


deviceQuery = DeviceQuery()
