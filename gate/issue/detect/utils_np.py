# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

DETECT UTILS with NP ONLY

"""

import numpy as np


def norm_boxes(boxes, shape):
  """Converts boxes from pixel coordinates to normalized coordinates.
  boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
  shape: [..., (height, width)] in pixels

  Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
  coordinates it's inside the box.

  Returns:
      [N, (y1, x1, y2, x2)] in normalized coordinates
  """
  h, w = shape
  scale = np.array([h - 1, w - 1, h - 1, w - 1])
  shift = np.array([0, 0, 1, 1])
  return np.divide((boxes - shift), scale).astype(np.float32)


def parse_image_meta_graph(meta):
  """Parses a tensor that contains image attributes to its components.
  See compose_image_meta() for more details.

  meta: [batch, meta length] where meta length depends on NUM_CLASSES

  Returns a dict of the parsed tensors.
  """
  image_id = meta[:, 0]
  original_image_shape = meta[:, 1:4]
  image_shape = meta[:, 4:7]
  window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
  scale = meta[:, 11]
  active_class_ids = meta[:, 12:]
  return {
      "image_id": image_id,
      "original_image_shape": original_image_shape,
      "image_shape": image_shape,
      "window": window,
      "scale": scale,
      "active_class_ids": active_class_ids,
  }
