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
"""GATE FRAMEWORK"""

import os
import sys
import argparse


def gate(args):
  """TENSORFLOW ENV"""
  # prepare for start tensorflow
  import tensorflow as tf
  tf.logging.set_verbosity(tf.logging.ERROR)
  # acquire dataset config file
  from gate.config.config_factory import load_config
  # get config for all settings.
  config = load_config(args)

  """Target"""
  if config.target == 'vision.classification':
    from samples.vision.classification import Classification as App
  elif config.target == 'vision.regression':
    from samples.vision.regression import Regression as App
  elif config.target.startswith('detect'):
    from samples.detect.detect import select as App
  elif config.target.startswith('trafficflow'):
    from samples.trafficflow.trafficflow import select as App
  elif config.target.startswith('avec2014'):
    from samples.avec2014.avec2014 import select as App
  elif config.target.startswith('kinface'):
    from samples.kinface.kinface import select as App
  else:
    raise ValueError('Unknown target [%s]' % config.target)

  """Instance"""
  Instance = App(config)

  """Task"""
  if config.task == 'train':
    Instance.train()
  elif config.task == 'test':
    Instance.test()
  elif config.task == 'val':
    Instance.val()
  elif config.task == 'heatmap':
    Instance.heatmap()
  else:
    raise ValueError('Unknown task [%s]' % config.task)


def run(args):
  """Initialize ENV"""
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  sys.path.append('gate/net')

  """Distribute the task"""
  if args.drawer is not None:
    from drawer import drawer
    drawer.interface(args.drawer)
  else:
    gate(args)


if __name__ == "__main__":
  # parse command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, dest='dataset', default=None)
  parser.add_argument('-gpu', type=str, dest='gpu', default='0')
  parser.add_argument('-config', type=str, dest='config', default=None)
  parser.add_argument('-task', type=str, dest='task', default=None)
  parser.add_argument('-model', type=str, dest='model', default=None)
  parser.add_argument('-drawer', type=str, dest='drawer', default=None)
  args, _ = parser.parse_known_args()
  run(args)
