# Copyright 2017 The KaiJIN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""2018/2/25 AVEC2014 """

import tensorflow as tf
from gate import context
from gate.data.data_factory import load_data
from gate.net.net_factory import net_graph
from gate.solver import updater
from gate.layer import l2
from gate.utils import variable
from gate.utils import filesystem
from gate.utils import string
from gate.utils.logger import logger
from gate.utils.heatmap import HeatMap
from samples.avec2014.utils import get_accurate_from_file


class AVEC2014_IMG_4VIEW(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)
