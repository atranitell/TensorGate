# -*- coding: utf-8 -*-
"""
DECTRION FRAMEWORK

Copyright (c) 2017 Kai JIN.
Licensed under the MIT License (see LICENSE for details)
Written by Kai JIN
Updated on 2018/4/19

--------------------------------------------------------

UTILS

"""

import random
import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform
import tensorflow as tf

import warnings

def compute_overlaps(boxes1, boxes2):
  """Computes IoU overlaps between two sets of boxes.
  boxes1, boxes2: [N, (y1, x1, y2, x2)].

  For better performance, pass the largest set first and the smaller second.
  """
  # Areas of anchors and GT boxes
  area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
  area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

  # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
  # Each cell contains the IoU value.
  overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
  for i in range(overlaps.shape[1]):
    box2 = boxes2[i]
    overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
  return overlaps


def box_refinement(box, gt_box):
  """Compute refinement needed to transform box to gt_box.
  box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
  assumed to be outside the box.
  """
  box = box.astype(np.float32)
  gt_box = gt_box.astype(np.float32)

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
  dh = np.log(gt_height / height)
  dw = np.log(gt_width / width)

  return np.stack([dy, dx, dh, dw], axis=1)


def compute_iou(box, boxes, box_area, boxes_area):
  """Calculates IoU of the given box with the array of the given boxes.
  box: 1D vector [y1, x1, y2, x2]
  boxes: [boxes_count, (y1, x1, y2, x2)]
  box_area: float. the area of 'box'
  boxes_area: array of length boxes_count.

  Note: the areas are passed in rather than calculated here for
        efficency. Calculate once in the caller to avoid duplicate work.
  """
  # Calculate intersection areas
  y1 = np.maximum(box[0], boxes[:, 0])
  y2 = np.minimum(box[2], boxes[:, 2])
  x1 = np.maximum(box[1], boxes[:, 1])
  x2 = np.minimum(box[3], boxes[:, 3])
  intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
  union = box_area + boxes_area[:] - intersection[:]
  iou = intersection / union
  return iou


def minimize_mask(bbox, mask, mini_shape):
  """Resize masks to a smaller version to reduce memory load.
  Mini-masks can be resized back to image scale using expand_masks()

  See inspect_data.ipynb notebook for more details.
  """
  mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
  for i in range(mask.shape[-1]):
    # Pick slice and cast to bool in case load_mask() returned wrong dtype
    m = mask[:, :, i].astype(bool)
    y1, x1, y2, x2 = bbox[i][:4]
    m = m[y1:y2, x1:x2]
    if m.size == 0:
      raise Exception("Invalid bounding box with area of zero")
    # Resize with bilinear interpolation
    m = skimage.transform.resize(m, mini_shape, order=1, mode="constant")
    mini_mask[:, :, i] = np.around(m).astype(np.bool)
  return mini_mask


def extract_bboxes(mask):
  """Compute bounding boxes from masks.
  mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

  Returns: bbox array [num_instances, (y1, x1, y2, x2)].
  """
  boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
  for i in range(mask.shape[-1]):
    m = mask[:, :, i]
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
      x1, x2 = horizontal_indicies[[0, -1]]
      y1, y2 = vertical_indicies[[0, -1]]
      # x2 and y2 should not be part of the box. Increment by 1.
      x2 += 1
      y2 += 1
    else:
      # No mask for this instance. Might happen due to
      # resizing or cropping. Set bbox to zeros
      x1, x2, y1, y2 = 0, 0, 0, 0
    boxes[i] = np.array([y1, x1, y2, x2])
  return boxes.astype(np.int32)


def resize_mask(mask, scale, padding, crop=None):
  """Resizes a mask using the given scale and padding.
  Typically, you get the scale and padding from resize_image() to
  ensure both, the image and the mask, are resized consistently.

  scale: mask scaling factor
  padding: Padding to add to the mask in the form
          [(top, bottom), (left, right), (0, 0)]
  """
  # Suppress warning from scipy 0.13.0, the output shape of zoom() is
  # calculated with round() instead of int()
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
  if crop is not None:
    y, x, h, w = crop
    mask = mask[y:y + h, x:x + w]
  else:
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
  return mask


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
  """Resizes an image keeping the aspect ratio unchanged.

  min_dim: if provided, resizes the image such that it's smaller
      dimension == min_dim
  max_dim: if provided, ensures that the image longest side doesn't
      exceed this value.
  min_scale: if provided, ensure that the image is scaled up by at least
      this percent even if min_dim doesn't require it.
  mode: Resizing mode.
      none: No resizing. Return the image unchanged.
      square: Resize and pad with zeros to get a square image
          of size [max_dim, max_dim].
      pad64: Pads width and height with zeros to make them multiples of 64.
             If min_dim or min_scale are provided, it scales the image up
             before padding. max_dim is ignored in this mode.
             The multiple of 64 is needed to ensure smooth scaling of feature
             maps up and down the 6 levels of the FPN pyramid (2**6=64).
      crop: Picks random crops from the image. First, scales the image based
            on min_dim and min_scale, then picks a random crop of
            size min_dim x min_dim. Can be used in training only.
            max_dim is not used in this mode.

  Returns:
  image: the resized image
  window: (y1, x1, y2, x2). If max_dim is provided, padding might
      be inserted in the returned image. If so, this window is the
      coordinates of the image part of the full image (excluding
      the padding). The x2, y2 pixels are not included.
  scale: The scale factor used to resize the image
  padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
  """
  # Keep track of image dtype and return results in the same dtype
  image_dtype = image.dtype
  # Default window (y1, x1, y2, x2) and default scale == 1.
  h, w = image.shape[:2]
  window = (0, 0, h, w)
  scale = 1
  padding = [(0, 0), (0, 0), (0, 0)]
  crop = None

  if mode == "none":
    return image, window, scale, padding, crop

  # Scale?
  if min_dim:
    # Scale up but not down
    scale = max(1, min_dim / min(h, w))
  if min_scale and scale < min_scale:
    scale = min_scale

  # Does it exceed max dim?
  if max_dim and mode == "square":
    image_max = max(h, w)
    if round(image_max * scale) > max_dim:
      scale = max_dim / image_max

  # Resize image using bilinear interpolation
  if scale != 1:
    image = skimage.transform.resize(
        image, (round(h * scale), round(w * scale)),
        order=1, mode="constant", preserve_range=True)

  # Need padding or cropping?
  if mode == "square":
    # Get new height and width
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
  elif mode == "pad64":
    h, w = image.shape[:2]
    # Both sides must be divisible by 64
    assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
    # Height
    if h % 64 > 0:
      max_h = h - (h % 64) + 64
      top_pad = (max_h - h) // 2
      bottom_pad = max_h - h - top_pad
    else:
      top_pad = bottom_pad = 0
    # Width
    if w % 64 > 0:
      max_w = w - (w % 64) + 64
      left_pad = (max_w - w) // 2
      right_pad = max_w - w - left_pad
    else:
      left_pad = right_pad = 0
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
  elif mode == "crop":
    # Pick a random crop
    h, w = image.shape[:2]
    y = random.randint(0, (h - min_dim))
    x = random.randint(0, (w - min_dim))
    crop = (y, x, min_dim, min_dim)
    image = image[y:y + min_dim, x:x + min_dim]
    window = (0, 0, min_dim, min_dim)
  else:
    raise Exception("Mode {} not supported".format(mode))
  return image.astype(image_dtype), window, scale, padding, crop


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
  """Takes attributes of an image and puts them in one 1D array.

  image_id: An int ID of the image. Useful for debugging.
  original_image_shape: [H, W, C] before resizing or padding.
  image_shape: [H, W, C] after resizing and padding
  window: (y1, x1, y2, x2) in pixels. The area of the image where the real
          image is (excluding the padding)
  scale: The scaling factor applied to the original image (float32)
  active_class_ids: List of class_ids available in the dataset from which
      the image came. Useful if training on images from multiple datasets
      where not all classes are present in all datasets.
  """
  meta = np.array(
      [image_id] +                  # size=1
      list(original_image_shape) +  # size=3
      list(image_shape) +           # size=3
      # size=4 (y1, x1, y2, x2) in image cooredinates
      list(window) +
      [scale] +                     # size=1
      list(active_class_ids)        # size=num_classes
  )
  return meta


def load_image_gt(dataset, config, image_id, augmentation=None,
                  use_mini_mask=False):
  """Load and return ground truth data for an image (image, mask, bounding boxes).

  augment: (Depricated. Use augmentation instead). If true, apply random
      image augmentation. Currently, only horizontal flipping is offered.
  augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
      For example, passing imgaug.augmenters.Fliplr(0.5) flips images
      right/left 50% of the time.
  use_mini_mask: If False, returns full-size masks that are the same height
      and width as the original image. These can be big, for example
      1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
      224x224 and are generated by extracting the bounding box of the
      object and resizing it to MINI_MASK_SHAPE.

  Returns:
  image: [height, width, 3]
  shape: the original shape of the image before resizing and cropping.
  class_ids: [instance_count] Integer class IDs
  bbox: [instance_count, (y1, x1, y2, x2)]
  mask: [height, width, instance_count]. The height and width are those
      of the image unless use_mini_mask is True, in which case they are
      defined in MINI_MASK_SHAPE.
  """
  # Load image and mask
  image = dataset.load_image(image_id)
  mask, class_ids = dataset.load_mask(image_id)
  original_shape = image.shape
  image, window, scale, padding, crop = resize_image(
      image,
      min_dim=config.IMAGE_MIN_DIM,
      min_scale=config.IMAGE_MIN_SCALE,
      max_dim=config.IMAGE_MAX_DIM,
      mode=config.IMAGE_RESIZE_MODE)
  mask = resize_mask(mask, scale, padding, crop)

  # Augmentation
  # This requires the imgaug lib (https://github.com/aleju/imgaug)
  if augmentation:
    import imgaug

    # Augmentors that are safe to apply to masks
    # Some, such as Affine, have settings that make them unsafe, so always
    # test your augmentation on masks
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]

    def hook(images, augmenter, parents, default):
      """Determines which augmenters to apply to masks."""
      return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

    # Store shapes before augmentation to compare
    image_shape = image.shape
    mask_shape = mask.shape
    # Make augmenters deterministic to apply similarly to images and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    # Change mask to np.uint8 because imgaug doesn't support np.bool
    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))
    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    # Change mask back to bool
    mask = mask.astype(np.bool)

  # Note that some boxes might be all zeros if the corresponding mask got cropped out.
  # and here is to filter them out
  _idx = np.sum(mask, axis=(0, 1)) > 0
  mask = mask[:, :, _idx]
  class_ids = class_ids[_idx]
  # Bounding boxes. Note that some boxes might be all zeros
  # if the corresponding mask got cropped out.
  # bbox: [num_instances, (y1, x1, y2, x2)]
  bbox = extract_bboxes(mask)

  # Active classes
  # Different datasets have different classes, so track the
  # classes supported in the dataset of this image.
  active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
  source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
  active_class_ids[source_class_ids] = 1

  # Resize masks to smaller size to reduce memory usage
  if use_mini_mask:
    mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

  # Image meta data
  image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                  window, scale, active_class_ids)

  return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
  """Generate targets for training Stage 2 classifier and mask heads.
  This is not used in normal training. It's useful for debugging or to train
  the Mask RCNN heads without using the RPN head.

  Inputs:
  rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
  gt_class_ids: [instance count] Integer class IDs
  gt_boxes: [instance count, (y1, x1, y2, x2)]
  gt_masks: [height, width, instance count] Grund truth masks. Can be full
            size or mini-masks.

  Returns:
  rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
  class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
  bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
          bbox refinements.
  masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
         to bbox boundaries and resized to neural network output size.
  """
  assert rpn_rois.shape[0] > 0
  assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
      gt_class_ids.dtype)
  assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
      gt_boxes.dtype)
  assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
      gt_masks.dtype)

  # It's common to add GT Boxes to ROIs but we don't do that here because
  # according to XinLei Chen's paper, it doesn't help.

  # Trim empty padding in gt_boxes and gt_masks parts
  instance_ids = np.where(gt_class_ids > 0)[0]
  assert instance_ids.shape[0] > 0, "Image must contain instances."
  gt_class_ids = gt_class_ids[instance_ids]
  gt_boxes = gt_boxes[instance_ids]
  gt_masks = gt_masks[:, :, instance_ids]

  # Compute areas of ROIs and ground truth boxes.
  rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
      (rpn_rois[:, 3] - rpn_rois[:, 1])
  gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
      (gt_boxes[:, 3] - gt_boxes[:, 1])

  # Compute overlaps [rpn_rois, gt_boxes]
  overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
  for i in range(overlaps.shape[1]):
    gt = gt_boxes[i]
    overlaps[:, i] = compute_iou(
        gt, rpn_rois, gt_box_area[i], rpn_roi_area)

  # Assign ROIs to GT boxes
  rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
  rpn_roi_iou_max = overlaps[np.arange(
      overlaps.shape[0]), rpn_roi_iou_argmax]
  # GT box assigned to each ROI
  rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
  rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

  # Positive ROIs are those with >= 0.5 IoU with a GT box.
  fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

  # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
  # TODO: To hard example mine or not to hard example mine, that's the question
#     bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
  bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

  # Subsample ROIs. Aim for 33% foreground.
  # FG
  fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
  if fg_ids.shape[0] > fg_roi_count:
    keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
  else:
    keep_fg_ids = fg_ids
  # BG
  remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
  if bg_ids.shape[0] > remaining:
    keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
  else:
    keep_bg_ids = bg_ids
  # Combine indicies of ROIs to keep
  keep = np.concatenate([keep_fg_ids, keep_bg_ids])
  # Need more?
  remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
  if remaining > 0:
    # Looks like we don't have enough samples to maintain the desired
    # balance. Reduce requirements and fill in the rest. This is
    # likely different from the Mask RCNN paper.

    # There is a small chance we have neither fg nor bg samples.
    if keep.shape[0] == 0:
      # Pick bg regions with easier IoU threshold
      bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
      assert bg_ids.shape[0] >= remaining
      keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
      assert keep_bg_ids.shape[0] == remaining
      keep = np.concatenate([keep, keep_bg_ids])
    else:
      # Fill the rest with repeated bg rois.
      keep_extra_ids = np.random.choice(
          keep_bg_ids, remaining, replace=True)
      keep = np.concatenate([keep, keep_extra_ids])
  assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
      "keep doesn't match ROI batch size {}, {}".format(
          keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

  # Reset the gt boxes assigned to BG ROIs.
  rpn_roi_gt_boxes[keep_bg_ids, :] = 0
  rpn_roi_gt_class_ids[keep_bg_ids] = 0

  # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
  rois = rpn_rois[keep]
  roi_gt_boxes = rpn_roi_gt_boxes[keep]
  roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
  roi_gt_assignment = rpn_roi_iou_argmax[keep]

  # Class-aware bbox deltas. [y, x, log(h), log(w)]
  bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                     config.NUM_CLASSES, 4), dtype=np.float32)
  pos_ids = np.where(roi_gt_class_ids > 0)[0]
  bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = box_refinement(
      rois[pos_ids], roi_gt_boxes[pos_ids, :4])
  # Normalize bbox refinements
  bboxes /= config.BBOX_STD_DEV

  # Generate class-specific target masks
  masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                   dtype=np.float32)
  for i in pos_ids:
    class_id = roi_gt_class_ids[i]
    assert class_id > 0, "class id must be greater than 0"
    gt_id = roi_gt_assignment[i]
    class_mask = gt_masks[:, :, gt_id]

    if config.USE_MINI_MASK:
      # Create a mask placeholder, the size of the image
      placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
      # GT box
      gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
      gt_w = gt_x2 - gt_x1
      gt_h = gt_y2 - gt_y1
      # Resize mini mask to size of GT box
      placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
          np.round(skimage.transform.resize(
              class_mask, (gt_h, gt_w), order=1, mode="constant")).astype(bool)
      # Place the mini batch in the placeholder
      class_mask = placeholder

    # Pick part of the mask and resize it
    y1, x1, y2, x2 = rois[i].astype(np.int32)
    m = class_mask[y1:y2, x1:x2]
    mask = skimage.transform.resize(
        m, config.MASK_SHAPE, order=1, mode="constant")
    masks[i, :, :, class_id] = mask

  return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
  """Given the anchors and GT boxes, compute overlaps and identify positive
  anchors and deltas to refine them to match their corresponding GT boxes.

  anchors: [num_anchors, (y1, x1, y2, x2)]
  gt_class_ids: [num_gt_boxes] Integer class IDs.
  gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

  Returns:
  rpn_match: [N] (int32) matches between anchors and GT boxes.
             1 = positive anchor, -1 = negative anchor, 0 = neutral
  rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
  """
  # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
  rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
  # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
  rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

  # Handle COCO crowds
  # A crowd box in COCO is a bounding box around several instances. Exclude
  # them from training. A crowd box is given a negative class ID.
  crowd_ix = np.where(gt_class_ids < 0)[0]
  if crowd_ix.shape[0] > 0:
    # Filter out crowds from ground truth class IDs and boxes
    non_crowd_ix = np.where(gt_class_ids > 0)[0]
    crowd_boxes = gt_boxes[crowd_ix]
    gt_class_ids = gt_class_ids[non_crowd_ix]
    gt_boxes = gt_boxes[non_crowd_ix]
    # Compute overlaps with crowd boxes [anchors, crowds]
    crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
    crowd_iou_max = np.amax(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)
  else:
    # All anchors don't intersect a crowd
    no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

  # Compute overlaps [num_anchors, num_gt_boxes]
  overlaps = compute_overlaps(anchors, gt_boxes)

  # Match anchors to GT Boxes
  # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
  # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
  # Neutral anchors are those that don't match the conditions above,
  # and they don't influence the loss function.
  # However, don't keep any GT box unmatched (rare, but happens). Instead,
  # match it to the closest anchor (even if its max IoU is < 0.3).
  #
  # 1. Set negative anchors first. They get overwritten below if a GT box is
  # matched to them. Skip boxes in crowd areas.
  anchor_iou_argmax = np.argmax(overlaps, axis=1)
  anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
  rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
  # 2. Set an anchor for each GT box (regardless of IoU value).
  # TODO: If multiple anchors have the same IoU match all of them
  gt_iou_argmax = np.argmax(overlaps, axis=0)
  rpn_match[gt_iou_argmax] = 1
  # 3. Set anchors with high overlap as positive.
  rpn_match[anchor_iou_max >= 0.7] = 1

  # Subsample to balance positive and negative anchors
  # Don't let positives be more than half the anchors
  ids = np.where(rpn_match == 1)[0]
  extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
  if extra > 0:
    # Reset the extra ones to neutral
    ids = np.random.choice(ids, extra, replace=False)
    rpn_match[ids] = 0
  # Same for negative proposals
  ids = np.where(rpn_match == -1)[0]
  extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                      np.sum(rpn_match == 1))
  if extra > 0:
    # Rest the extra ones to neutral
    ids = np.random.choice(ids, extra, replace=False)
    rpn_match[ids] = 0

  # For positive anchors, compute shift and scale needed to transform them
  # to match the corresponding GT boxes.
  ids = np.where(rpn_match == 1)[0]
  ix = 0  # index into rpn_bbox
  # TODO: use box_refinement() rather than duplicating the code here
  for i, a in zip(ids, anchors[ids]):
    # Closest gt box (it might have IoU < 0.7)
    gt = gt_boxes[anchor_iou_argmax[i]]

    # Convert coordinates to center plus width/height.
    # GT Box
    gt_h = gt[2] - gt[0]
    gt_w = gt[3] - gt[1]
    gt_center_y = gt[0] + 0.5 * gt_h
    gt_center_x = gt[1] + 0.5 * gt_w
    # Anchor
    a_h = a[2] - a[0]
    a_w = a[3] - a[1]
    a_center_y = a[0] + 0.5 * a_h
    a_center_x = a[1] + 0.5 * a_w

    # Compute the bbox refinement that the RPN should predict.
    rpn_bbox[ix] = [
        (gt_center_y - a_center_y) / a_h,
        (gt_center_x - a_center_x) / a_w,
        np.log(gt_h / a_h),
        np.log(gt_w / a_w),
    ]
    # Normalize
    rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
    ix += 1

  return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
  """Generates ROI proposals similar to what a region proposal network
  would generate.

  image_shape: [Height, Width, Depth]
  count: Number of ROIs to generate
  gt_class_ids: [N] Integer ground truth class IDs
  gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

  Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
  """
  # placeholder
  rois = np.zeros((count, 4), dtype=np.int32)

  # Generate random ROIs around GT boxes (90% of count)
  rois_per_box = int(0.9 * count / gt_boxes.shape[0])
  for i in range(gt_boxes.shape[0]):
    gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
    h = gt_y2 - gt_y1
    w = gt_x2 - gt_x1
    # random boundaries
    r_y1 = max(gt_y1 - h, 0)
    r_y2 = min(gt_y2 + h, image_shape[0])
    r_x1 = max(gt_x1 - w, 0)
    r_x2 = min(gt_x2 + w, image_shape[1])

    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
      y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
      x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
      # Filter out zero area boxes
      threshold = 1
      y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                  threshold][:rois_per_box]
      x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                  threshold][:rois_per_box]
      if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
        break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    box_rois = np.hstack([y1, x1, y2, x2])
    rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

  # Generate random ROIs anywhere in the image (10% of count)
  remaining_count = count - (rois_per_box * gt_boxes.shape[0])
  # To avoid generating boxes with zero area, we generate double what
  # we need and filter out the extra. If we get fewer valid boxes
  # than we need, we loop and try again.
  while True:
    y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
    x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
    # Filter out zero area boxes
    threshold = 1
    y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                threshold][:remaining_count]
    x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                threshold][:remaining_count]
    if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
      break

  # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
  # into x1, y1, x2, y2 order
  x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
  y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
  global_rois = np.hstack([y1, x1, y2, x2])
  rois[-remaining_count:] = global_rois
  return rois
