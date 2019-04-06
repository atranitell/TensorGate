# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from gate.net.deepfuse import resnet_utils
from gate.env import env
from gate.utils.logger import logger

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  """Bottleneck residual unit variant with BN before convolutions.

  This is the full preactivation residual unit variant proposed in [2]. See
  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
  variant which has an extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v2_vanilla(inputs,
                      blocks,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      include_root_block=True,
                      spatial_squeeze=True,
                      reuse=None,
                      scope=None):
  with tf.variable_scope(scope + '_aux', 'resnet_v2_aux', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError(
                  'The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        end_points['gap_conv'] = net
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='logits_aux')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points


def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
  data = tf.unstack(inputs, axis=1)
  aux_inputs = data[0]  # RGB
  inputs = data[1]  # Optical Flow
  aux_logit, aux_net = resnet_v2_vanilla(
      aux_inputs, blocks, num_classes, is_training,
      global_pool, output_stride, include_root_block,
      spatial_squeeze, reuse, scope)

  with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d, bottleneck,
                         resnet_utils.stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError(
                  'The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(
            net, blocks, output_stride, aux_net=aux_net)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        end_points['gap_conv'] = net
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
          end_points['global_pool'] = net

        if env.target == 'avec2014.img.bicnn.shared':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.shared')
          net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                           tf.squeeze(end_points['global_pool'], [1, 2]),
                           aux_logit], axis=1)
          net = tf.contrib.layers.fully_connected(
              net, num_classes,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits')
          end_points['flow_logit'] = net
          end_points['rgb_logit'] = aux_logit

        elif env.target == 'avec2014.img.bicnn.2shared':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.2shared')
          net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                           tf.squeeze(end_points['global_pool'], [1, 2]),
                           aux_logit], axis=1)
          net = tf.contrib.layers.fully_connected(
              net, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits')
          end_points['flow_logit'] = net
          end_points['rgb_logit'] = aux_logit

        elif env.target == 'avec2014.img.bicnn.orth':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth')
          # rgb orth [n, 2048]
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          rgb_s, rgb_p = tf.split(rgb_feat, axis=1, num_or_size_splits=2)
          l_rgb_orth = tf.reduce_mean(tf.reduce_sum(rgb_s * rgb_p, axis=1))

          # flow orth [n, 2048]
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])
          flow_s, flow_p = tf.split(flow_feat, axis=1, num_or_size_splits=2)
          l_flow_orth = tf.reduce_mean(tf.reduce_sum(flow_s * flow_p, axis=1))

          # distribution loss
          l_dist_mean = tf.nn.l2_loss(tf.reduce_mean(rgb_s - flow_s, axis=0))
          bs_avg_rgb_s = tf.pow(tf.reduce_mean(rgb_s, axis=0), 2)
          bs_avg_flow_s = tf.pow(tf.reduce_mean(flow_s, axis=0), 2)
          bs_rgb_s = tf.reduce_sum(rgb_s * rgb_s, axis=0) / 31.0
          bs_flow_s = tf.reduce_sum(flow_s * flow_s, axis=0) / 31.0
          l_dist_std = 2 * tf.nn.l2_loss(bs_rgb_s - bs_avg_rgb_s +
                                         bs_flow_s - bs_avg_flow_s)
          l_dist = l_dist_mean + l_dist_std

          # prediction
          net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                           tf.squeeze(end_points['global_pool'], [1, 2]),
                           aux_logit], axis=1)
          net = tf.contrib.layers.fully_connected(
              net, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits')

          end_points['flow_logit'] = net
          end_points['rgb_logit'] = aux_logit
          end_points['l_rgb_orth'] = l_rgb_orth
          end_points['l_flow_orth'] = l_flow_orth
          end_points['l_dist'] = l_dist

        elif env.target == 'avec2014.img.bicnn.orth2':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth2')
          # rgb orth [n, 2048]
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          rgb_s, rgb_p = tf.split(rgb_feat, axis=1, num_or_size_splits=2)
          l_rgb_orth = tf.reduce_mean(tf.reduce_sum(rgb_s * rgb_p, axis=1))

          # flow orth [n, 2048]
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])
          flow_s, flow_p = tf.split(flow_feat, axis=1, num_or_size_splits=2)
          l_flow_orth = tf.reduce_mean(tf.reduce_sum(flow_s * flow_p, axis=1))

          # modal orth
          l_modal_orth = tf.reduce_mean(tf.reduce_sum(rgb_p * flow_p, axis=1))
          share = tf.concat([rgb_s, flow_s], axis=1)

          net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                           tf.squeeze(end_points['global_pool'], [1, 2]),
                           aux_logit], axis=1)
          net = tf.contrib.layers.fully_connected(
              net, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits',
              reuse=False)

        elif env.target == 'avec2014.img.bicnn.orth2a':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth2a')
          # rgb orth [n, 2048]
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          rgb_s, rgb_p = tf.split(rgb_feat, axis=1, num_or_size_splits=2)
          l_rgb_orth = tf.reduce_mean(tf.reduce_sum(rgb_s * rgb_p, axis=1))

          # flow orth [n, 2048]
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])
          flow_s, flow_p = tf.split(flow_feat, axis=1, num_or_size_splits=2)
          l_flow_orth = tf.reduce_mean(tf.reduce_sum(flow_s * flow_p, axis=1))

          # modal orth
          l_modal_orth = tf.reduce_mean(tf.reduce_sum(rgb_p * flow_p, axis=1))
          share = tf.concat([rgb_s, flow_s], axis=1)

        elif env.target == 'avec2014.img.bicnn.orth3':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth3')
          # rgb orth [n, 2048]
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          rgb_s, rgb_p = tf.split(rgb_feat, axis=1, num_or_size_splits=2)
          l_rgb_orth = tf.reduce_mean(tf.reduce_sum(rgb_s * rgb_p, axis=1))

          # flow orth [n, 2048]
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])
          flow_s, flow_p = tf.split(flow_feat, axis=1, num_or_size_splits=2)
          l_flow_orth = tf.reduce_mean(tf.reduce_sum(flow_s * flow_p, axis=1))

          # modal orth
          l_modal_orth = tf.reduce_mean(tf.reduce_sum(rgb_p * flow_p, axis=1))
          # share merge
          share = tf.concat([rgb_s, flow_s], axis=1)
          # share output
          share_logit = tf.contrib.layers.fully_connected(
              share, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits_share',
              reuse=False)

          # flow merge
          net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                           tf.squeeze(end_points['global_pool'], [1, 2]),
                           aux_logit], axis=1)
          # flow output
          net = tf.contrib.layers.fully_connected(
              net, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits',
              reuse=False)

          end_points['flow_logit'] = net
          end_points['rgb_logit'] = aux_logit
          end_points['share_logit'] = share_logit
          end_points['l_rgb_orth'] = l_rgb_orth
          end_points['l_flow_orth'] = l_flow_orth
          end_points['l_madal_orth'] = l_modal_orth

        elif env.target == 'avec2014.img.bicnn.orth4':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth4')

          def Transform(feat, dims):
            """feat [N, C]"""
            with tf.variable_scope('transformer'):
              W = tf.Variable(tf.random_normal(
                  [1, dims], stddev=0.01))
              X = W * feat
            return X, W

          # backbone output
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])

          # transform1-rgb
          rgb_p_feat_t, w_rgb_p = Transform(rgb_feat, 2048)
          rgb_s_feat_t, w_rgb_s = Transform(rgb_feat, 2048)
          l_rgb_ps = 0.5 * tf.reduce_sum(tf.pow(w_rgb_p * w_rgb_s, 2))

          # transform2-rgb
          rgb_m_feat = tf.concat([rgb_p_feat_t, rgb_s_feat_t], axis=1)
          rgb_m_feat, w_rgb_m = Transform(rgb_m_feat, 4096)

          # regression-rgb
          rgb_logit = tf.contrib.layers.fully_connected(
              rgb_m_feat, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits_rgb',
              reuse=False)

          # [2048], [2048]
          rgb_s_mean, rgb_s_var = tf.nn.moments(rgb_s_feat_t, axes=0)

          # transform1-flow
          flow_p_feat_t, w_flow_p = Transform(flow_feat, 2048)
          flow_s_feat_t, w_flow_s = Transform(flow_feat, 2048)
          l_flow_ps = 0.5 * tf.reduce_sum(tf.pow(w_flow_p * w_flow_s, 2))
          flow_s_mean, flow_s_var = tf.nn.moments(flow_s_feat_t, axes=0)

          # transform2-flow
          flow_m_feat = tf.concat([flow_p_feat_t, flow_s_feat_t], axis=1)
          flow_m_feat, w_flow_m = Transform(flow_m_feat, 4096)

          # regression-flow
          out_feat = tf.concat([rgb_m_feat, flow_m_feat, rgb_logit], axis=1)
          out_logit = tf.squeeze(tf.contrib.layers.fully_connected(
              out_feat, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits_out',
              reuse=False))

          # share part
          l_s_mean = 0.5 * tf.reduce_sum(
              tf.pow(rgb_s_mean-flow_s_mean, 2))
          l_s_std = 0.5 * tf.reduce_sum(
              tf.pow(tf.sqrt(rgb_s_var)-tf.sqrt(flow_s_var), 2))
          l_s = l_s_mean + l_s_std
          l_f = tf.reduce_sum(tf.pow(w_rgb_m * w_flow_m, 2))

          end_points['flow_logit'] = out_logit
          end_points['rgb_logit'] = rgb_logit
          end_points['l_rgb_orth'] = l_rgb_ps
          end_points['l_flow_orth'] = l_flow_ps
          end_points['l_share_orth'] = l_s
          end_points['l_trace'] = l_f

        elif env.target == 'avec2014.img.bicnn.orth5':
          logger.info('resnet_v2_bishared by avec2014.img.bicnn.orth5')

          def Transform(feat, dims, name, reuse=None):
            """feat [N, C]"""
            with tf.variable_scope('transformer', reuse=reuse):
              W = tf.get_variable(
                  name,
                  shape=[1, dims],
                  initializer=tf.truncated_normal_initializer(stddev=0.01))
              X = W * feat
            return X, W

          # backbone output
          rgb_feat = tf.squeeze(aux_net['global_pool'], [1, 2])
          flow_feat = tf.squeeze(end_points['global_pool'], [1, 2])

          # transform1-rgb
          rgb_p_feat_t, w_rgb_p = Transform(rgb_feat, 2048, 'rgb_p')
          rgb_s_feat_t, w_rgb_s = Transform(rgb_feat, 2048, 'share')
          l_rgb_ps = 0.5 * tf.reduce_sum(tf.pow(w_rgb_p * w_rgb_s, 2))

          # transform2-rgb
          rgb_m_feat = tf.concat([rgb_p_feat_t, rgb_s_feat_t], axis=1)
          rgb_m_feat, w_rgb_m = Transform(rgb_m_feat, 4096, 'rgb_m')

          # regression-rgb
          rgb_logit = tf.contrib.layers.fully_connected(
              rgb_m_feat, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits_rgb',
              reuse=False)

          # [2048], [2048]
          rgb_s_mean, rgb_s_var = tf.nn.moments(rgb_s_feat_t, axes=0)

          # transform1-flow
          flow_p_feat_t, w_flow_p = Transform(flow_feat, 2048, 'flow_p')
          flow_s_feat_t, w_flow_s = Transform(flow_feat, 2048, 'share', True)
          l_flow_ps = 0.5 * tf.reduce_sum(tf.pow(w_flow_p * w_flow_s, 2))
          flow_s_mean, flow_s_var = tf.nn.moments(flow_s_feat_t, axes=0)

          # transform2-flow
          flow_m_feat = tf.concat([flow_p_feat_t, flow_s_feat_t], axis=1)
          flow_m_feat, w_flow_m = Transform(flow_m_feat, 4096, 'flow_m')

          # regression-flow
          out_feat = tf.concat([rgb_m_feat, flow_m_feat, rgb_logit], axis=1)
          out_logit = tf.squeeze(tf.contrib.layers.fully_connected(
              out_feat, 1,
              biases_initializer=None,
              weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
              weights_regularizer=None,
              activation_fn=None,
              scope='logits_out',
              reuse=False))

          # share part
          l_s_mean = 0.5 * tf.reduce_sum(
              tf.pow(rgb_s_mean-flow_s_mean, 2))
          l_s_std = 0.5 * tf.reduce_sum(
              tf.pow(tf.sqrt(rgb_s_var)-tf.sqrt(flow_s_var), 2))
          l_s = l_s_mean + l_s_std
          l_f = tf.reduce_sum(tf.pow(w_rgb_m * w_flow_m, 2))

          end_points['flow_logit'] = out_logit
          end_points['rgb_logit'] = rgb_logit
          end_points['l_rgb_orth'] = l_rgb_ps
          end_points['l_flow_orth'] = l_flow_ps
          end_points['l_share_orth'] = l_s
          end_points['l_trace'] = l_f

        else:
          raise ValueError('Unknown Error')

  if env.target == 'avec2014.img.bicnn.orth2':
    with tf.variable_scope(scope+'_aux', 'resnet_v2_aux', auxiliary_name_scope=False):
      share = tf.reshape(share, [-1, 1, 1, 2048])
      share_logit = slim.conv2d(share, 1, [1, 1], activation_fn=None,
                                normalizer_fn=None, scope='logits_aux', reuse=True)
      share_logit = tf.squeeze(share_logit, [1, 2])
      end_points['flow_logit'] = net
      end_points['rgb_logit'] = aux_logit
      end_points['share_logit'] = share_logit
      end_points['l_rgb_orth'] = l_rgb_orth
      end_points['l_flow_orth'] = l_flow_orth
      end_points['l_madal_orth'] = l_modal_orth

  elif env.target == 'avec2014.img.bicnn.orth2a':
    with tf.variable_scope(scope+'_aux', 'resnet_v2_aux', auxiliary_name_scope=False):
      share = tf.reshape(share, [-1, 1, 1, 2048])
      share_logit = slim.conv2d(share, 1, [1, 1], activation_fn=None,
                                normalizer_fn=None, scope='logits_aux', reuse=True)
      share_logit = tf.squeeze(share_logit, [1, 2])

    with tf.variable_scope(scope, 'resnet_v2'):
      net = tf.concat([tf.squeeze(aux_net['global_pool'], [1, 2]),
                       tf.squeeze(end_points['global_pool'], [1, 2]),
                       aux_logit,
                       share_logit], axis=1)
      net = tf.contrib.layers.fully_connected(
          net, 1,
          biases_initializer=None,
          weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
          weights_regularizer=None,
          activation_fn=None,
          scope='logits',
          reuse=False)

      end_points['flow_logit'] = net
      end_points['rgb_logit'] = aux_logit
      end_points['share_logit'] = share_logit
      end_points['l_rgb_orth'] = l_rgb_orth
      end_points['l_flow_orth'] = l_flow_orth
      end_points['l_madal_orth'] = l_modal_orth

  return net, end_points


resnet_v2.default_image_size = 224


def resnet_v2_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v2 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v2 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


resnet_v2.default_image_size = 224


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v2_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


resnet_v2_50.default_image_size = resnet_v2.default_image_size


def resnet_v2_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_101'):
  """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


resnet_v2_101.default_image_size = resnet_v2.default_image_size


def resnet_v2_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_152'):
  """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


resnet_v2_152.default_image_size = resnet_v2.default_image_size


def resnet_v2_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v2_200'):
  """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
  blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


resnet_v2_200.default_image_size = resnet_v2.default_image_size
