# -*- coding: utf-8 -*-
""" path helper
    updated: 2017/11/25
"""
import os


class Path():
  """ Path operation
  """

  def join_step(self, dst, step, fmt='txt', ext=''):
    """ format: dst + '/' + '%8d' + ext + '.' + fmt
    """
    return os.path.join(dst, '%08d%s.%s' % (int(step), ext, fmt))

  def join(self, dst, ext, fmt='txt'):
    """ format: dst + '/' + ext + '.' + fmt
    """
    return os.path.join(dst, '%s.%s' % (ext, fmt))

  def filename(self, abspath):
    """ acquire filename with extension of a path
      automatically transfer to str type.
    Input: /p1/p2/f1.ext
    Return: f1.ext
    """
    if type(abspath) is not str:
      return os.path.split(str(abspath, encoding="utf-8"))[1]
    return os.path.split(abspath)[1]

  def join_name(self, dst, src):
    """ e.g.
      dst = '/home/kj/tensorflow/'
      src = '/home/kj/gate/test.txt'
      ret: '/home/kj/tensorflow/text.txt'
    """
    return os.path.join(dst, self.filename(src))


path = Path()
