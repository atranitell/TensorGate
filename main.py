# -*- coding: utf-8 -*-
"""
GATE FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

Computer Vision for image recognition covering:
- Classification + Localizaiton
- Object Detection
- Semantic Segmentation
- Instance Segmentation

"""

import os
import sys
import argparse

GPU_ID = '0' if len(sys.argv) == 1 else sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('gate/net')

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from gate.config.factory import get_config


def run(dataset, config_file):
  # get config for all settings.
  config = get_config(dataset, config_file)

  """ Target """
  if config.target == 'cnn.classification':
    from gate.issue.cnn.classification import Classification as App
  elif config.target == 'cnn.regression':
    from gate.issue.cnn.regression import Regression as App
  elif config.target.startswith('detect'):
    from gate.issue.detect.detect import select as App
  elif config.target.startswith('trafficflow'):
    from gate.issue.trafficflow.trafficflow import select as App
  elif config.target.startswith('avec2014'):
    from gate.issue.avec2014.avec2014 import select as App
  elif config.target.startswith('kinface'):
    from gate.issue.kinface.kinface import select as App

  """ Task """
  if config.task == 'train':
    App(config).train()
  elif config.task == 'test':
    App(config).test()
  elif config.task == 'val':
    App(config).val()
  elif config.task == 'heatmap':
    App(config).heatmap()


if __name__ == "__main__":
  # parse command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, dest='dataset', default='coco2014')
  parser.add_argument('-config', type=str, dest='config_file', default=None)
  args, _ = parser.parse_known_args()
  run(args.dataset, args.config_file)
