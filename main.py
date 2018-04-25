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

import argparse


def run

if __name__ == "__main__":
  # parse command line
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, dest='dataset')
  parser.add_argument('-config', type=str, dest='config', default=None)
  args, _ = parser.parse_known_args()
