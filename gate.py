# -*- coding: utf-8 -*-
""" All interface to access the framework
    Author: Kai JIN
    Updated: 2017/11/19
"""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from task import task


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-config', type=str, dest='config',
                      default='_datasets/settings.json')
  args, _ = parser.parse_known_args()

  if not os.path.exists(args.config):
    raise ValueError('Configuration file path %s is invalid.' % args.config)

  task.initilize(args.config)
  task.run()