# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/21

--------------------------------------------------------

DETECT UTILS

"""

import tensorflow as tf


def norm_boxes_graph(boxes, height, width):
  """ Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    height: image height in pixels
    width: image width in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
  """
  scale = tf.to_float(tf.stack([height, width, height, width], axis=-1))
  scale = scale - tf.constant(1.0)
  shift = tf.constant([0., 0., 1., 1.])
  return tf.divide(boxes - shift, scale)


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
  """ Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
  """
  if not isinstance(inputs, list):
    inputs = [inputs]

  outputs = []
  for i in range(batch_size):
    inputs_slice = [x[i] for x in inputs]
    output_slice = graph_fn(*inputs_slice)
    if not isinstance(output_slice, (tuple, list)):
      output_slice = [output_slice]
    outputs.append(output_slice)
  # Change outputs from a list of slices where each is
  # a list of outputs to a list of outputs and each has
  # a list of slices
  outputs = list(zip(*outputs))

  if names is None:
    names = [None] * len(outputs)

  result = [tf.stack(o, axis=0, name=n)
            for o, n in zip(outputs, names)]
  if len(result) == 1:
    result = result[0]

  return result


def apply_box_deltas_graph(boxes, deltas):
  """ Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
  """
  # Convert to y, x, h, w
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  center_y = boxes[:, 0] + 0.5 * height
  center_x = boxes[:, 1] + 0.5 * width
  # Apply deltas
  center_y += deltas[:, 0] * height
  center_x += deltas[:, 1] * width
  height *= tf.exp(deltas[:, 2])
  width *= tf.exp(deltas[:, 3])
  # Convert back to y1, x1, y2, x2
  y1 = center_y - 0.5 * height
  x1 = center_x - 0.5 * width
  y2 = y1 + height
  x2 = x1 + width
  result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
  return result


def clip_boxes_graph(boxes, window):
  """ boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
  """
  # Split
  wy1, wx1, wy2, wx2 = tf.split(window, 4)
  y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
  # Clip
  y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
  x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
  y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
  x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
  clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
  clipped.set_shape((clipped.shape[0], 4))
  return clipped


def box_refinement_graph(box, gt_box):
  """Compute refinement needed to transform box to gt_box.
  box and gt_box are [N, (y1, x1, y2, x2)]
  """
  box = tf.cast(box, tf.float32)
  gt_box = tf.cast(gt_box, tf.float32)

  height = box[:, 2] - box[:, 0]
  width = box[:, 3] - box[:, 1]
  center_y = box[:, 0] + 0.5 * height
  center_x = box[:, 1] + 0.5 * width

  gt_height = gt_box[:, 2] - gt_box[:, 0]
  gt_width = gt_box[:, 3] - gt_box[:, 1]
  gt_center_y = gt_box[:, 0] + 0.5 * gt_height
  gt_center_x = gt_box[:, 1] + 0.5 * gt_width

  dy = (gt_center_y - center_y) / height
  dx = (gt_center_x - center_x) / width
  dh = tf.log(gt_height / height)
  dw = tf.log(gt_width / width)

  result = tf.stack([dy, dx, dh, dw], axis=1)
  return result


def overlaps_graph(boxes1, boxes2):
  """Computes IoU overlaps between two sets of boxes.
  boxes1, boxes2: [N, (y1, x1, y2, x2)].
  """
  # 1. Tile boxes2 and repeate boxes1. This allows us to compare
  # every boxes1 against every boxes2 without loops.
  # TF doesn't have an equivalent to np.repeate() so simulate it
  # using tf.tile() and tf.reshape.
  b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                          [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
  b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
  # 2. Compute intersections
  b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
  b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
  y1 = tf.maximum(b1_y1, b2_y1)
  x1 = tf.maximum(b1_x1, b2_x1)
  y2 = tf.minimum(b1_y2, b2_y2)
  x2 = tf.minimum(b1_x2, b2_x2)
  intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
  # 3. Compute unions
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area + b2_area - intersection
  # 4. Compute IoU and reshape to [boxes1, boxes2]
  iou = intersection / union
  overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
  return overlaps


def trim_zeros_graph(boxes, name=None):
  """Often boxes are represented with matricies of shape [N, 4] and
  are padded with zeros. This removes zero boxes.

  boxes: [N, 4] matrix of boxes.
  non_zeros: [N] a 1D boolean mask identifying the rows to keep
  """
  non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
  boxes = tf.boolean_mask(boxes, non_zeros, name=name)
  return boxes, non_zeros


def log2_graph(x):
  """Implementatin of Log2. TF doesn't have a native implemenation."""
  return tf.log(x) / tf.log(2.0)
