""" Task Set for kinface
"""
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT
from issue.kinface.kinvae_bidirect2 import KINVAE_BIDIRECT2
from issue.kinface.kinvae_encoder import KINVAE_ENCODER
from issue.kinface.kinvae_encoder2 import KINVAE_ENCODER2
from issue.kinface.kinvae_feature import KINVAE_FEATURE


def select(config):
  """ select different subtask
  """
  if config.target == 'kinvae.encoder':
    return KINVAE_ENCODER(config)
  elif config.target == 'kinvae.encoder2':
    return KINVAE_ENCODER2(config)
  elif config.target == 'kinvae.bidirect':
    return KINVAE_BIDIRECT(config)
  elif config.target == 'kinvae.bidirect2':
    return KINVAE_BIDIRECT2(config)
  elif config.target == 'kinvae.feature':
    return KINVAE_FEATURE(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
