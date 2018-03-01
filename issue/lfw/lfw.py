""" Task Set for LFW
"""
from issue.lfw.lfw_vae import LFW_VAE


def select(config):
  """ select different subtask
  """
  if config.target == 'lfw.vae':
    return LFW_VAE(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
