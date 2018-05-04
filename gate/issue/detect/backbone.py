
# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

BACKBONE COLLECTIONS

"""

from gate.net.factory import get_net
from gate.layer import ops


def resnet_50(x, config, phase):
  _, backbone_map = get_net(x, config.net[0], phase)
  C2 = backbone_map['resnet_v2_50/block1/unit_2/bottleneck_v2']
  C3 = backbone_map['resnet_v2_50/block2/unit_3/bottleneck_v2']
  C4 = backbone_map['resnet_v2_50/block3/unit_5/bottleneck_v2']
  C5 = backbone_map['resnet_v2_50/block4/unit_3/bottleneck_v2']
  return [C2, C3, C4, C5]
