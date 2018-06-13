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
from datetime import datetime


def init_config(arguments):
  """Load Config"""
  # acquire dataset config file
  from gate.config.config_factory import load_config
  # get config for all settings.
  config = load_config(arguments)
  # there, if the command args has extra parameters, it will rewrite it.
  config.rewrite_command_args()
  # returns
  return config


def init_logger(config):
  """ initialize logger """
  from gate.utils import filesystem
  from gate.utils import string
  from gate.utils.logger import logger
  from gate.env import env
  pid = datetime.strftime(datetime.now(), '%y%m%d%H%M%S')
  # if output_dir is None, to make a new dir to save model
  # else we use the value of output_dir as workspace
  filename = string.join_dots(config.name, pid)
  if config.output_dir is None:
    config.output_dir = filesystem.mkdir(env._OUTPUT + filename)
  else:
    # if the folder is not exisited, but output_dir has been assigned.
    # the routinue will create a folder
    filesystem.mkdir(config.output_dir)
  logger.init(filename, config.output_dir)
  logger.info('Initilized logger successful.')
  logger.info('Current model in %s' % config.output_dir)


def gate(arguments):
  """init env"""
  config = init_config(arguments)
  init_logger(config)

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

  """Task"""
  if config.task == 'train':
    App(config).train()
  elif config.task == 'test':
    App(config).test()
  elif config.task == 'val':
    App(config).val()
  elif config.task == 'heatmap':
    App(config).heatmap()
  else:
    raise ValueError('Unknown task [%s]' % config.task)


def run(arguments):
  """Initialize ENV"""
  os.environ["CUDA_VISIBLE_DEVICES"] = arguments.gpu
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  """Distribute the task"""
  if arguments.drawer is not None:
    from drawer import drawer
    drawer.interface(arguments.drawer)
  else:
    sys.path.append('gate/net')
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    gate(arguments)


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

  if args.dataset is None and args.drawer is None:
    print('The input is NULL')
    exit(1)

  run(args)
