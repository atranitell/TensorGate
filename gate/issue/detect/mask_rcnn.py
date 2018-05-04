# -*- coding: utf-8 -*-
"""
Gate FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

MASK RCNN Implementation

The design is as follows:

... -> RoI ----\
                -> RoIFeatureXform -> mask head -> mask output -> loss
... -> Feature /
       Map

The mask head produces a feature representation of the RoI for the purpose
of mask prediction. The mask output module converts the feature representation
into real-valued (soft) masks.

"""

import numpy as np
import tensorflow as tf
from gate import context
from gate.solver import updater
from gate.util import variable
from gate.util import checkpoint
from gate.data.factory import get_data
from gate.issue import detect


class MASK_RCNN(context.Context):

  def __init__(self, config):
    context.Context.__init__(self, config)
    self.cfg = detect.mask_rcnn_config.Config()
    self.config.cfg = self.cfg

  def _backbone(self):
    """ [C2, C3, C4, C5] """
    self.backbone_maps = detect.backbone.resnet_50(
        self.input_image, self.config, self.phase)

  def _fpn(self):
    """ [P2, P3, P4, P5, P6] """
    self.fpn_feature_maps = detect.fpn.fpn_resnet_graph(self.backbone_maps)
    # [P2, P3, P4, P5, P6]
    self.rpn_feature_maps = self.fpn_feature_maps
    # [P2, P3, P4, P5]
    self.mrcnn_feature_maps = self.fpn_feature_maps[:-1]

  def _anchors(self):
    """ only using for training phase.
      Generate multi-scale crops according to input shape.
      Due to the mask-rcnn using FCN, we need generate a series of anchors
        for different scale feature maps.
    """
    self.fpn_shapes = [m.get_shape().as_list()[1:3]
                       for m in self.rpn_feature_maps]
    anchors = detect.anchors.generate_pyramid_anchors(
        self.cfg.RPN_ANCHOR_SCALES,
        self.cfg.RPN_ANCHOR_RATIOS,
        self.fpn_shapes,
        self.cfg.BACKBONE_STRIDES,
        self.cfg.RPN_ANCHOR_STRIDE)
    anchors = np.broadcast_to(anchors, (self.cfg.BATCH_SIZE,) + anchors.shape)
    # [Batchsize, Anchors, 4]
    self.input_anchors = tf.convert_to_tensor(detect.utils_np.norm_boxes(
        anchors, self.cfg.IMAGE_SHAPE[:2]))

  def _rpn(self):
    """ extract feature patch from each hierachy.
      the patch shape is identical with the anchors
    """
    rpn_maps = detect.rpn.rpn_graph(
        feature_maps=self.rpn_feature_maps,
        anchors_per_location=len(self.cfg.RPN_ANCHOR_RATIOS),
        anchor_stride=self.cfg.RPN_ANCHOR_STRIDE)
    # [batchsize, num_anchors, 2] (BG/FG)
    # [batchsize, num_anchors, 2] (BG/FG)
    # [batchsize, num_anchors, 4] ((dy, dx, log(dh), log(dw))
    rpns_name = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    rpns = list(zip(*rpn_maps))
    rpns = [tf.concat(o, axis=1, name=n) for o, n in zip(rpns, rpns_name)]
    self.rpn_class_logits, self.rpn_class, self.rpn_bbox = rpns
    # proposal to select limited bbox
    proposal_count = self.cfg.POST_NMS_ROIS_TRAINING \
        if self.is_training else self.cfg.POST_NMS_ROIS_INFERENCE
    # by nms to reduce proposal bbox and merge bbox
    self.rpn_rois = detect.rpn.generate_proposal(
        rpn_probs=self.rpn_class,
        rpn_bbox=self.rpn_bbox,
        anchors=self.input_anchors,
        proposal_count=proposal_count,
        cfg=self.cfg)

  def _detector(self):
    # Generate detection targets
    # Subsamples proposals and generates target outputs for training
    # Note that proposal class IDs, gt_boxes, and gt_masks are zero
    # padded. Equally, returned rois and targets are zero padded.
    outputs = detect.detector.detection_head_graph(
        proposals=self.rpn_rois,
        gt_class_ids=self.input_gt_class_ids,
        gt_boxes=self.input_gt_boxes,
        gt_masks=self.input_gt_masks,
        config=self.cfg)
    self.rois, self.target_class_ids, self.target_bbox, self.target_mask = outputs

  def _fpn_classifier_head(self):
    outpus = detect.fpn.fpn_classifier_graph(
        rois=self.rois,
        feature_maps=self.mrcnn_feature_maps,
        image_meta=self.input_image_meta,
        pool_size=self.cfg.POOL_SIZE,
        num_classes=self.cfg.NUM_CLASSES,
        is_training=self.is_training)
    self.mrcnn_class_logits, self.mrcnn_class, self.mrcnn_bbox = outpus

  def _fpn_mask_head(self):
    """ mrcnn_mask: [batch, roi_count, height, width, num_classes]
    """
    self.mrcnn_mask = detect.fpn.fpn_mask_graph(
        rois=self.rois,
        feature_maps=self.mrcnn_feature_maps,
        image_meta=self.input_image_meta,
        pool_size=self.cfg.MASK_POOL_SIZE,
        num_classes=self.cfg.NUM_CLASSES,
        is_training=self.is_training)

  def _rpn_class_loss(self):
    self.rpn_class_loss = detect.rpn.rpn_class_loss_graph(
        self.input_rpn_match, self.rpn_class_logits)

  def _rpn_bbox_loss(self):
    self.rpn_bbox_loss = detect.rpn.rpn_bbox_loss_graph(
        self.cfg, self.input_rpn_bbox, self.input_rpn_match, self.rpn_bbox)

  def _mrcnn_class_loss(self):
    active_class_ids = detect.utils_np.parse_image_meta_graph(
        self.input_image_meta)['active_class_ids']
    self.class_loss = detect.fpn.mrcnn_class_loss_graph(
        self.target_class_ids, self.mrcnn_class_logits, active_class_ids)

  def _mrcnn_bbox_loss(self):
    self.bbox_loss = detect.fpn.mrcnn_bbox_loss_graph(
        self.target_bbox, self.target_class_ids, self.mrcnn_bbox)

  def _mrcnn_mask_loss(self):
    self.mask_loss = detect.fpn.mrcnn_mask_loss_graph(
        self.target_mask, self.target_class_ids, self.mrcnn_mask)

  def _loss(self):
    self._rpn_class_loss()
    self._rpn_bbox_loss()
    self._mrcnn_class_loss()
    self._mrcnn_bbox_loss()
    self._mrcnn_mask_loss()
    self.loss = self.rpn_class_loss + self.rpn_bbox_loss + \
        self.class_loss + self.bbox_loss + self.mask_loss

  def _placeholder(self):
    """ prepare input data
    """
    # preparing for training and test
    h, w, c = self.cfg.IMAGE_SHAPE
    self.input_image = tf.placeholder(
        tf.float32, [None, h, w, c], 'input_image')
    self.input_image_meta = tf.placeholder(
        tf.float32, [None, self.cfg.IMAGE_META_SIZE], 'input_image_meta')
    # preparing for training
    if self.is_training:
      # RPN GT
      self.input_rpn_match = tf.placeholder(
          tf.int32, [None, None, 1], 'input_rpn_match')
      self.input_rpn_bbox = tf.placeholder(
          tf.float32, [None, None, 4], 'input_rpn_bbox')
      # Detection GT (class IDs, bounding boxes, and masks)
      # 1. GT Class IDs (zero padded)
      self.input_gt_class_ids = tf.placeholder(
          tf.int32, [None, None], 'input_gt_class_ids')
      # 2. GT Boxes in pixels (zero padded)
      # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
      self.input_gt_boxes = tf.placeholder(
          tf.float32, [None, None, 4], 'input_gt_boxes')
      # Normalize coordinates
      self.gt_boxes = detect.utils.norm_boxes_graph(self.input_gt_boxes, h,  w)
      # mini-mask
      if self.cfg.USE_MINI_MASK:
        mini_h, mini_w = self.cfg.MINI_MASK_SHAPE
        self.input_gt_masks = tf.placeholder(
            tf.bool, [None, mini_h, mini_w,  None], 'input_gt_masks')
      else:
        self.input_gt_masks = tf.placeholder(
            tf.bool, [None, h, w, None], 'input_gt_masks')
    else:
      self.input_anchors = tf.placeholder(
          tf.float32, [None, None, 4], 'input_anchors')

  def train(self):
    self._enter_('train')
    with tf.Graph().as_default() as graph:
      # load data
      dataset = get_data(self.config)
      # setting placeholder
      self._placeholder()
      # backbone
      self._backbone()
      # feature pyramid network
      self._fpn()
      # generate input anchors - train only
      self._anchors()
      # regeon proposal network
      self._rpn()
      # detector
      self._detector()
      # fpn head
      self._fpn_classifier_head()
      self._fpn_mask_head()

      # loss functions
      self._loss()

      # layers = 'heads'
      # layer_regex = {
      #     'heads': r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
      #     # From a specific Resnet stage and up
      #     "3+": r"(resnet_v2_50/block2.*)|(resnet_v2_50/block3.*)|(resnet_v2_50/block4.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
      #     "4+": r"(resnet_v2_50/block3.*)|(resnet_v2_50/block4.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
      #     "5+": r"(resnet_v2_50/block4.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
      #     # All layers
      #     "all": ".*"
      # }
      # if layers in layer_regex.keys():
      #   layers = layer_regex[layers]

      var_list_op, var_list = checkpoint.ckpt_read(
          '../_models/resnet_v2_50/ckpt/resnet_v2_50.ckpt')

      global_step = tf.train.create_global_step()
      _vars = variable.exclude_vars('resnet_v2_50')
      train_op = updater.default(self.config, self.loss, global_step, _vars)

      # add hooks
      self.add_hook(self.snapshot.init())
      self.add_hook(self.summary.init())
      self.add_hook(context.Running_Hook(
          config=self.config.log,
          step=global_step,
          keys=['loss'],
          values=[self.loss],
          func_test=None,
          func_val=None))

      dataset.init(
          backbone_shapes=self.fpn_shapes,
          anchors=detect.anchors.generate_pyramid_anchors(
              self.cfg.RPN_ANCHOR_SCALES,
              self.cfg.RPN_ANCHOR_RATIOS,
              self.fpn_shapes,
              self.cfg.BACKBONE_STRIDES,
              self.cfg.RPN_ANCHOR_STRIDE))

      batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = dataset.next()
      feed_dict = {
          self.input_image: batch_images,
          self.input_image_meta: batch_image_meta,
          self.input_rpn_match: batch_rpn_match,
          self.input_rpn_bbox: batch_rpn_bbox,
          self.input_gt_class_ids: batch_gt_class_ids,
          self.input_gt_boxes: batch_gt_boxes,
          self.input_gt_masks: batch_gt_masks}

      # var_list = var_list.update(feed_dict)
      # for key in feed_dict:
      #   var_list.update()
      for key in feed_dict:
        var_list[key] = feed_dict[key]

      saver = tf.train.Saver(var_list=variable.all())
      with context.DefaultSession(self.hooks) as sess:
        self.snapshot.restore(sess, saver)
        sess.run(var_list_op, var_list)
        while not sess.should_stop():
          batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = dataset.next()
          feed_dict = {
              self.input_image: batch_images,
              self.input_image_meta: batch_image_meta,
              self.input_rpn_match: batch_rpn_match,
              self.input_rpn_bbox: batch_rpn_bbox,
              self.input_gt_class_ids: batch_gt_class_ids,
              self.input_gt_boxes: batch_gt_boxes,
              self.input_gt_masks: batch_gt_masks}
          sess.run(train_op, feed_dict=feed_dict)

    self._exit_()

  def val(self):
    self._enter_('val')
    with tf.Graph().as_default() as graph:
      dataset = get_data(self.config)
    self._exit_()
