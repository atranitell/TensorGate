""" Task Set for kinface
"""
from issue.kinface.kinvae_bidirect import KINVAE_BIDIRECT
from issue.kinface.kinvae_bidirect2 import KINVAE_BIDIRECT2
from issue.kinface.kinvae_bidirect3 import KINVAE_BIDIRECT3
from issue.kinface.kinvae_bidirect4 import KINVAE_BIDIRECT4
from issue.kinface.kinvae_bidirect5 import KINVAE_BIDIRECT5
from issue.kinface.kinvae_bidirect6 import KINVAE_BIDIRECT6
from issue.kinface.kinvae_bidirect7 import KINVAE_BIDIRECT7
from issue.kinface.kinvae_bidirect8 import KINVAE_BIDIRECT8
from issue.kinface.kinvae_bidirect9 import KINVAE_BIDIRECT9
from issue.kinface.kinvae_bidirect10 import KINVAE_BIDIRECT10
from issue.kinface.kinvae_bidirect11 import KINVAE_BIDIRECT11

from issue.kinface.kinvae_encoder import KINVAE_ENCODER
from issue.kinface.kinvae_encoder2 import KINVAE_ENCODER2
from issue.kinface.kinvae_encoder3 import KINVAE_ENCODER3
from issue.kinface.kinvae_encoder4 import KINVAE_ENCODER4

from issue.kinface.kinvae_feature import KINVAE_FEATURE


def select(config):
  """ select different subtask
  """
  if config.target == 'kinvae.encoder':
    return KINVAE_ENCODER(config)
  elif config.target == 'kinvae.encoder2':
    return KINVAE_ENCODER2(config)
  elif config.target == 'kinvae.encoder3':
    return KINVAE_ENCODER3(config)
  elif config.target == 'kinvae.encoder4':
    return KINVAE_ENCODER4(config)
  elif config.target == 'kinvae.bidirect':
    return KINVAE_BIDIRECT(config)
  elif config.target == 'kinvae.bidirect2':
    return KINVAE_BIDIRECT2(config)
  elif config.target == 'kinvae.bidirect3':
    return KINVAE_BIDIRECT3(config)
  elif config.target == 'kinvae.bidirect4':
    return KINVAE_BIDIRECT4(config)
  elif config.target == 'kinvae.bidirect5':
    return KINVAE_BIDIRECT5(config)
  elif config.target == 'kinvae.bidirect6':
    return KINVAE_BIDIRECT6(config)
  elif config.target == 'kinvae.bidirect7':
    return KINVAE_BIDIRECT7(config)
  elif config.target == 'kinvae.bidirect8':
    return KINVAE_BIDIRECT8(config)
  elif config.target == 'kinvae.bidirect9':
    return KINVAE_BIDIRECT9(config)
  elif config.target == 'kinvae.bidirect10':
    return KINVAE_BIDIRECT10(config)
  elif config.target == 'kinvae.bidirect11':
    return KINVAE_BIDIRECT11(config)
  elif config.target == 'kinvae.feature':
    return KINVAE_FEATURE(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
