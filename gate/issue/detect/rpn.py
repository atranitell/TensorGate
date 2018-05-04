
# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/23

--------------------------------------------------------

Region Proposal Network

"""

import numpy as np
import keras.backend as K
from gate.layer.ops import *
from gate.issue.detect import utils


def generate_proposal(rpn_probs,
                      rpn_bbox,
                      anchors,
                      proposal_count,
                      cfg):
  """Receives anchor scores and selects a subset to pass as proposals
  to the second stage. Filtering is done based on anchor scores and
  non-max suppression to remove overlaps. It also applies bounding
  box refinement deltas to anchors.

  Inputs:
      rpn_probs: [batch, anchors, (bg prob, fg prob)]
      rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
      anchors: [batch, (y1, x1, y2, x2)] anchors in normalized coordinates

  Returns:
      Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
  """
  pre_max_nms_limit = 6000
  # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
  scores = rpn_probs[:, :, 1]
  # box deltas [batch, num_rois, 4]
  deltas = rpn_bbox
  deltas *= tf.reshape(tf.to_float(cfg.RPN_BBOX_STD_DEV), [-1, 1, 4])

  # Improve performance by trimming to top anchors by score
  # and doing the rest on the smaller subset.
  pre_nms_limit = tf.minimum(pre_max_nms_limit, tf.shape(anchors)[1])
  ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                   name="top_anchors").indices
  scores = utils.batch_slice(
      [scores, ix], lambda x, y: tf.gather(x, y), cfg.BATCH_SIZE)
  deltas = utils.batch_slice(
      [deltas, ix], lambda x, y: tf.gather(x, y), cfg.BATCH_SIZE)
  pre_nms_anchors = tf.to_float(utils.batch_slice(
      [anchors, ix], lambda a, x: tf.gather(a, x), cfg.BATCH_SIZE,
      names=["pre_nms_anchors"]))

  # Apply deltas to anchors to get refined anchors.
  # [batch, N, (y1, x1, y2, x2)]
  boxes = utils.batch_slice(
      [pre_nms_anchors, deltas],
      lambda x, y: utils.apply_box_deltas_graph(x, y), cfg.BATCH_SIZE,
      names=["refined_anchors"])

  # Clip to image boundaries. Since we're in normalized coordinates,
  # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
  window = np.array([0, 0, 1, 1], dtype=np.float32)
  boxes = utils.batch_slice(
      boxes, lambda x: utils.clip_boxes_graph(x, window), cfg.BATCH_SIZE,
      names=["refined_anchors_clipped"])

  # Filter out small boxes
  # According to Xinlei Chen's paper, this reduces detection accuracy
  # for small objects, so we're skipping it.

  # Non-max suppression
  def nms(boxes, scores):
    indices = tf.image.non_max_suppression(
        boxes, scores, proposal_count,
        cfg.RPN_NMS_THRESHOLD, name="rpn_non_max_suppression")
    proposals = tf.gather(boxes, indices)
    # Pad if needed
    padding = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
    proposals = tf.pad(proposals, [(0, padding), (0, 0)])
    return proposals

  proposals = utils.batch_slice([boxes, scores], nms, cfg.BATCH_SIZE)
  proposals = tf.reshape(proposals, [-1, proposal_count, 4])
  return proposals


def rpn_graph(feature_maps, anchors_per_location, anchor_stride):
  """ Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                  every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H, W, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
  """
  n_maps = len(feature_maps)
  rpn_maps = []
  for i in range(n_maps):
    reuse = False if i == 0 else True
    shared_conv = relu(conv2d(feature_maps[i], 512, 3, stride=anchor_stride,
                              name='rpn_shared_conv', reuse=reuse))
    rpn_maps.append(sub_rpn_graph(
        shared_conv, anchors_per_location, scope='rpn_%d' % (i+1)))
  return rpn_maps


def sub_rpn_graph(feature_map, anchors_per_location, scope, reuse=None):
  """ the rest of rpn part.
  """
  with tf.variable_scope(scope):
    H, W, C = feature_map.get_shape().as_list()[1:]
    n_anchors = H*W*anchors_per_location
    # [batch, height, width, anchors per location * 2]
    rpn_class_raw = conv2d(
        feature_map, 2*anchors_per_location, 3, 1, name='rpn_class_raw')
    # Reshape to [batch, anchors, 2]
    rpn_class_logits = tf.reshape(
        rpn_class_raw, [-1, int(n_anchors), 2], 'rpn_class_logits')
    # Softmax on last dimension of BG/FG.
    rpn_probs = layers.softmax(rpn_class_logits, scope='rpn_class_fbg')
    # Bounding box refinement. [batch, H, W, anchors per location, depth]
    # where depth is [x, y, log(w), log(h)]
    rpn_bbox_pred = conv2d(feature_map, anchors_per_location*4, 1, 1)
    # Reshape to [batch, anchors, 4]
    rpn_bbox = tf.reshape(rpn_bbox_pred, [-1, n_anchors, 4], 'rpn_bbox_pred')
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
  """RPN anchor classifier loss.

  rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
             -1=negative, 0=neutral anchor.
  rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
  """
  # Squeeze last dim to simplify
  rpn_match = tf.squeeze(rpn_match, -1)
  # Get anchor classes. Convert the -1/+1 match to 0/1 values.
  anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
  # Positive and Negative anchors contribute to the loss,
  # but neutral anchors (match value = 0) don't.
  indices = tf.where(K.not_equal(rpn_match, 0))
  # Pick rows that contribute to the loss and filter out the rest.
  rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
  anchor_class = tf.gather_nd(anchor_class, indices)
  # Crossentropy loss
  loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                           output=rpn_class_logits,
                                           from_logits=True)
  loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
  return loss


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
  """Return the RPN bounding box loss graph.

  config: the model config object.
  target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
      Uses 0 padding to fill in unsed bbox deltas.
  rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
             -1=negative, 0=neutral anchor.
  rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
  """
  # Positive anchors contribute to the loss, but negative and
  # neutral anchors (match value of 0 or -1) don't.
  rpn_match = K.squeeze(rpn_match, -1)
  indices = tf.where(K.equal(rpn_match, 1))

  # Pick bbox deltas that contribute to the loss
  rpn_bbox = tf.gather_nd(rpn_bbox, indices)

  # Trim target bounding box deltas to the same length as rpn_bbox.
  batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
  target_bbox = batch_pack_graph(target_bbox, batch_counts,
                                 config.IMAGES_PER_GPU)

  # TODO: use smooth_l1_loss() rather than reimplementing here
  #       to reduce code duplication
  diff = K.abs(target_bbox - rpn_bbox)
  less_than_one = K.cast(K.less(diff, 1.0), "float32")
  loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)

  loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
  return loss


def batch_pack_graph(x, counts, num_rows):
  """ Picks different number of values from each row in x depending 
    on the values in counts.
  """
  outputs = []
  for i in range(num_rows):
    outputs.append(x[i, :counts[i]])
  return tf.concat(outputs, axis=0)
