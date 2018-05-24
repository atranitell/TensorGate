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
"""Choose a task to execuate."""

from gate.config.dataset import mnist
from gate.config.dataset import trafficflow
from gate.config.dataset import kinface
from gate.config.dataset import avec2014
# from gate.config.dataset import coco

config_map = {
    'mnist': mnist.MNIST,
    'trafficflow': trafficflow.TRAFFICFLOW,
    'kinface.vae': kinface.KinfaceVAE,
    'avec2014': avec2014.AVEC2014,
    'avec2014.flow': avec2014.AVEC2014_FLOW,
    'avec2014.bicnn': avec2014.AVEC2014_BICNN,
    'avec2014.audio.cnn': avec2014.AVEC2014_AUDIO_CNN,
    'avec2014.audio.fcn': avec2014.AVEC2014_AUDIO_FCN
    # 'coco2014': coco.COCO2014
}


def load_config(args):
  """dataset config factory

  The module will check args and load config file, to specific output.
  raise error when args.dataset is not existed in config_map. 
  Also, args.task and args.model will rewrite the default model config.

  Args:
    args: the command from the input.

  Returns:
    a config class.
  """
  if args.dataset not in config_map:
    raise ValueError('Unknown dataset name %s' % args.dataset)
  return config_map[args.dataset](args)
