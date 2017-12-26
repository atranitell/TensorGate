""" task
"""
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT
from issue.kinface.kinvae_bidirect2 import KINVAE_BIDIRECT2
from issue.kinface.kinvae_bidirect3 import KINVAE_BIDIRECT3
from issue.kinface.kinvae_encoder import KINVAE_ENCODER
from issue.kinface.kinvae_feature import KINVAE_FEATURE
from issue.kinface.kinvae_encoder_2way import KINVAE_ENCODER2
from issue.kinface.kinvae_encoder_1l import KINVAE_ENCODER_1L


def select(config):
  """ select different subtask
  """
  if config.target == 'kinvae.encoder':
    return KINVAE_ENCODER(config)
  elif config.target == 'kinvae.encoder2':
    return KINVAE_ENCODER2(config)
  elif config.target == 'kinvae.encoder3':
    return KINVAE_ENCODER_1L(config)
  elif config.target == 'kinvae.bidirect':
    return KINVAE_BIDIRECT(config)
  elif config.target == 'kinvae.bidirect2':
    return KINVAE_BIDIRECT2(config)
  elif config.target == 'kinvae.bidirect3':
    return KINVAE_BIDIRECT3(config)
  elif config.target == 'kinvae.feature':
    return KINVAE_FEATURE(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
