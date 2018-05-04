# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

COCO DATABASE

"""

import os
import scipy
import tensorflow as tf
import numpy as np
from gate.data.custom import utils
from gate.util.logger import logger
from pycocotools import coco

import skimage.color
import skimage.io
import skimage.transform


class DB_COCO():

  def __init__(self, config):
    # load pre-config file
    self.config = config
    self.cfg = config.cfg
    self.coco = coco.COCO(self.config.data.entry_path)

    # store the data point
    self.image_info = []
    self.imgid_to_ix = {}
    self.class_info = [{'source': self.config.name, 'id': 0, 'name': 'BG'}]
    self.catid_to_ix = {}

  def _init_class(self):
    """ initialize dataset class label.
    """
    # class_ids is 81 categories with list(int) type.
    class_ids = sorted(self.coco.getCatIds())
    ix = 0
    for class_id in class_ids:
      # if the class has been added
      for info in self.class_info:
        if info['source'] == self.config.name and info['id'] == class_id:
          break
      # if not, add a new class
      self.class_info.append({
          'source': self.config.name,
          'id': class_id,
          'name': self.coco.loadCats(class_id)[0]['name']})
      self.catid_to_ix[class_id] = ix
      ix += 1

    # Map sources to class_ids they support
    self.sources = list(set([i['source'] for i in self.class_info]))
    self.source_class_ids = {}
    # Loop over datasets
    for source in self.sources:
      self.source_class_ids[source] = []
      # Find classes that belong to this dataset
      for i, info in enumerate(self.class_info):
        # Include BG class in all datasets
        if i == 0 or source == info['source']:
          self.source_class_ids[source].append(i)

  def _init_image(self):
    """ load image meta information.
    """
    def join(idx):
      filename = self.coco.imgs[idx]['file_name']
      data_dir = self.config.data.data_dir
      return os.path.join(data_dir, filename)

    class_ids = sorted(self.coco.getCatIds())
    image_ids = list(self.coco.imgs.keys())
    for ix, imgId in enumerate(image_ids):
      info = {
          'source': self.config.name,
          'id': imgId,
          'path': join(imgId),
          'width': self.coco.imgs[imgId]['width'],
          'height': self.coco.imgs[imgId]['height'],
          'annotations': self.coco.loadAnns(
              self.coco.getAnnIds(
                  imgIds=[imgId],
                  catIds=class_ids,
                  iscrowd=None))}
      self.image_info.append(info)
      self.imgid_to_ix[imgId] = ix

  def __next__(self):
    """ acquire the next batch.
    """
    print('hello')

  def __len__(self):
    """ acquire the number of data items.
    """
    return len(self.image_info)

  def __str__(self):
    """ print database information.
    """

  def __getitem__(self, n):
    """ acquire specific position elements.
    """

  def _annotation_to_mask(self, annotation, height, width):
    """ Convert annotation which can be polygons, uncompressed RLE,
        or RLE(run-length encoding) to binary mask.
    Return:
      binary mask (numpy 2D array)
    """
    segm = annotation['segmentation']
    if isinstance(segm, list):
      # polygon -- a single object might consist of multiple parts
      # we merge all parts into one mask rle code
      rles = coco.maskUtils.frPyObjects(segm, height, width)
      rle = coco.maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
      # uncompressed RLE
      rle = coco.maskUtils.frPyObjects(segm, height, width)
    else:
      # rle
      rle = annotation['segmentation']
    return coco.maskUtils.decode(rle)

  def load_mask(self, ix):
    """ Load instance masks for the given image.
      Consider a image including multi-entities, we need to distiguish them
      one by one. Thus, it requires us to construct a 3D [H, W, N] bitmap
      mask tensor and corresponding matrix [N] in terms of their classes id.

      Different datasets use different ways to store masks. This
      function converts the different mask format to one format
      in the form of a bitmap [height, width, instances].

      Returns:
      masks: A bool array of shape [height, width, instance count] with
          one mask per instance.
      class_ids: a 1D array of class IDs of the instance masks.
    """
    instance_masks = []
    class_ids = []
    image_info = self.image_info[ix]
    annotations = self.image_info[ix]['annotations']
    # Build mask of shape [height, width, instance_count] and list
    # of class IDs that correspond to each channel of the mask.
    for ann in annotations:
      class_id = self.catid_to_ix[ann['category_id']]
      mask = self._annotation_to_mask(
          ann, image_info['height'], image_info['width'])
      # Some objects are so small that they're less than 1 pixel area
      # and end up rounded out. Skip those objects.
      if mask.max() < 1:
        continue
      # Is it a crowd? If so, use a negative class ID.
      if ann['iscrowd']:
        # Use negative class ID for crowds
        class_id *= -1
        # For crowd masks, annToMask() sometimes returns a mask
        # smaller than the given dimensions. If so, resize it.
        _ms_h, _ms_w = mask.shape
        if _ms_h != image_info["height"] or _ms_w != image_info["width"]:
          mask = np.ones([image_info["height"], image_info['width']], bool)
      instance_masks.append(mask)
      class_ids.append(class_id)

    # pack instance masks into an array
    if class_ids:
      mask = np.stack(instance_masks, axis=2).astype(np.bool)
      class_ids = np.array(class_ids, dtype=np.int32)
    else:
      mask = np.empty([0, 0, 0])
      class_ids = np.empty([0], np.int32)

    return mask, class_ids

  def _mold_inputs(self, images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).

    If input image with shape (360, 640, 3)
      molded_image: (1024, 1024, 3)
      window: (224, 0, 800, 1024)
      scale: 1.6
      padding: [(224, 224), (0, 0), (0, 0)]
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
      # resize to square
      molded_image, window, scale, padding, crop = utils.resize_image(
          image,
          min_dim=self.cfg.IMAGE_MIN_DIM,
          max_dim=self.cfg.IMAGE_MAX_DIM,
          min_scale=self.cfg.IMAGE_MIN_SCALE,
          mode=self.cfg.IMAGE_RESIZE_MODE)
      # reduce mean pixel
      molded_image = image.astype(np.float32) - \
          np.array(self.cfg.MEAN_PIXEL)
      # compose image meta
      image_meta = utils.compose_image_meta(
          0, image.shape, molded_image.shape, window, scale,
          np.zeros([self.cfg.NUM_CLASSES], dtype=np.int32))
      # add in
      molded_images.append(molded_image)
      windows.append(window)
      image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows

  def load_image(self, ix):
    """ Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(self.image_info[ix]['path'])
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
      image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
      image = image[..., :3]
    return image

  @property
  def num_classes(self):
    return self.cfg.NUM_CLASSES

  def init(self, backbone_shapes, anchors):
    # initailize class
    self._init_class()
    logger.info('%d categories has been initialized.' % len(self.class_info))
    # initailize image
    self._init_image()
    logger.info('%d images has been initialized.' % len(self.image_info))

    self.backbone_shapes = backbone_shapes
    self.anchors = anchors

    self.iter = 0
    self.cur_idx = -1

  def next(self):
    # TODO: SHUFFLE
    # TODO: BATCHSIZE > 1 SITUATION

    # Skip images that have no instances. This can happen in cases
    # where we train on a subset of classes and the image doesn't
    # have any of the classes we care about.
    while (1):
      self.cur_idx += 1
      image, image_meta, gt_class_ids, gt_boxes, gt_masks = utils.load_image_gt(
          self, self.cfg, self.cur_idx, None, self.cfg.USE_MINI_MASK)

      if not np.any(gt_class_ids > 0):
        continue
      else:
        break

    rpn_match, rpn_bbox = utils.build_rpn_targets(
        image.shape, self.anchors, gt_class_ids, gt_boxes, self.cfg)

    # TODO random rois
    batch_image_meta = np.zeros(
        (self.cfg.BATCH_SIZE,) + image_meta.shape, dtype=image_meta.dtype)
    batch_rpn_match = np.zeros(
        [self.cfg.BATCH_SIZE, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
    batch_rpn_bbox = np.zeros(
        [self.cfg.BATCH_SIZE, self.cfg.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
    batch_images = np.zeros(
        (self.cfg.BATCH_SIZE,) + image.shape, dtype=np.float32)
    batch_gt_class_ids = np.zeros(
        (self.cfg.BATCH_SIZE, self.cfg.MAX_GT_INSTANCES), dtype=np.int32)
    batch_gt_boxes = np.zeros(
        (self.cfg.BATCH_SIZE, self.cfg.MAX_GT_INSTANCES, 4), dtype=np.int32)
    batch_gt_masks = np.zeros(
        (self.cfg.BATCH_SIZE, gt_masks.shape[0], gt_masks.shape[1],
         self.cfg.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

    if gt_boxes.shape[0] > self.cfg.MAX_GT_INSTANCES:
      ids = np.random.choice(
          np.arange(gt_boxes.shape[0]), self.cfg.MAX_GT_INSTANCES, replace=False)
      gt_class_ids = gt_class_ids[ids]
      gt_boxes = gt_boxes[ids]
      gt_masks = gt_masks[:, :, ids]

    # Add to batch
    batch_images[0] = image.astype(np.float32) - np.array(self.cfg.MEAN_PIXEL)
    batch_image_meta[0] = image_meta
    batch_rpn_match[0] = rpn_match[:, np.newaxis]
    batch_rpn_bbox[0] = rpn_bbox
    batch_gt_class_ids[0, :gt_class_ids.shape[0]] = gt_class_ids
    batch_gt_boxes[0, :gt_boxes.shape[0]] = gt_boxes
    batch_gt_masks[0, :, :, :gt_masks.shape[-1]] = gt_masks

    outputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
               batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
    return outputs

    self.iter += 1
