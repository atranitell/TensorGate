# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

FOR KINFACE SERIES

kinface.1E
  C |       |-> C_feat |
    |-> E-> |          |-> cosine
  P |       |-> P_feat |

kinface.2E
  C |-> E1-> |-> C_feat |
                        |-> cosine
  P |-> E2-> |-> P_feat |

kinface.1E1G
  C |       |-> C_feat |        | -> C_fake
    |-> E-> |          |-> G -> |
  P |       |-> P_feat |        | -> P_fake

kinface.1E1G1D        
                                     C_P_real |         
  C |       |-> C_feat |        | -> C_fake   |       | L_D_C
    |-> E-> |          |-> G -> |             |-> D ->|
  P |       |-> P_feat |        | -> P_fake   |       | L_D_P
                                     P_C_real |
kinface.1E2G2D
                                     C_P_real |         
  C |       |-> C_feat |-> G1 ->| -> C_fake   |-> D1 -> | L_D_C
    |-> E-> |          |        |             |        |
  P |       |-> P_feat |-> G2 ->| -> P_fake   |-> D2 -> | L_D_P
                                     P_C_real |

kinface.1E2G2D.ccpp
                                     C_real |         
  C |       |-> C_feat |-> G1 ->| -> C_fake   |-> D1 -> | L_D_C
    |-> E-> |          |        |             |        |
  P |       |-> P_feat |-> G2 ->| -> P_fake   |-> D2 -> | L_D_P
                                     P_real |

"""

from gate.issue.kinface.kinface_1E import KINFACE_1E
from gate.issue.kinface.kinface_2E import KINFACE_2E
from gate.issue.kinface.kinface_1E1G import KINFACE_1E1G
from gate.issue.kinface.kinface_1E1G1D import KINFACE_1E1G1D
from gate.issue.kinface.kinface_1E2G2D import KINFACE_1E2G2D
from gate.issue.kinface.kinface_1E2G2D_ccpp import KINFACE_1E2G2D_CCPP


def select(config):
  """ select different subtask
  """
  if config.target == 'kinface.1E':
    return KINFACE_1E(config)
  elif config.target == 'kinface.2E':
    return KINFACE_2E(config)
  elif config.target == 'kinface.1E1G':
    return KINFACE_1E1G(config)
  elif config.target == 'kinface.1E1G1D':
    return KINFACE_1E1G1D(config)
  elif config.target == 'kinface.1E2G2D':
    return KINFACE_1E2G2D(config)
  elif config.target == 'kinface.1E2G2D.ccpp':
    return KINFACE_1E2G2D_CCPP(config)
  else:
    raise ValueError('Unknown Target [%s]' % config.target)
