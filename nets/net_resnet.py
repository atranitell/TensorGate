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
Another difference is that 'v2' ResNets do not include an activation function in
the main pathway. Also see [2; Fig. 4e].

Typical use:

   from tensorflow.contrib.layers.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with arg_scope(resnet_v2.resnet_arg_scope(is_training)):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import utils

from nets import net
from nets import net_resnet_utils


class resent(net.Net):

    def __init__(self):
        self.weight_decay = 0.0001
        self.batch_norm_decay = 0.997
        self.batch_norm_epsilon = 1e-5
        self.batch_norm_scale = True
        self.global_pool = False
        self.output_stride = None
        self.resue = None
        self.include_root_block = True

    def resnet_v2(self, inputs, blocks, num_classes=None, is_training=True, scope=None):
        """Generator for v2 (preactivation) ResNet models.

        This function generates a family of ResNet v2 models. See the resnet_v2_*()
        methods for specific model instantiations, obtained by selecting different
        block instantiations that produce ResNets of various depths.

        Training for image classification on Imagenet is usually done with [224, 224]
        inputs, resulting in [7, 7] feature maps at the output of the last ResNet
        block for the ResNets defined in [1] that have nominal stride equal to 32.
        However, for dense prediction tasks we advise that one uses inputs with
        spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
        this case the feature maps at the ResNet output will have spatial shape
        [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
        and corners exactly aligned with the input image corners, which greatly
        facilitates alignment of the features to the image. Using as input [225, 225]
        images results in [8, 8] feature maps at the output of the last ResNet block.

        For dense prediction tasks, the ResNet needs to run in fully-convolutional
        (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
        have nominal stride equal to 32 and a good choice in FCN mode is to use
        output_stride=16 in order to increase the density of the computed features at
        small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

        Args:
            inputs: A tensor of size [batch, height_in, width_in, channels].
            blocks: A list of length equal to the number of ResNet blocks. Each element
            is a net_resnet_utils.Block object describing the units in the block.
            num_classes: Number of predicted classes for classification tasks. If None
            we return the features before the logit layer.
            is_training: whether is training or not.
            global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
            output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
            include_root_block: If True, include the initial convolution followed by
            max-pooling, if False excludes it. If excluded, `inputs` should be the
            results of an activation-less convolution.
            reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
            scope: Optional variable_scope.


        Returns:
            net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is None, then
            net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is not None, net contains the pre-softmax
            activations.
            end_points: A dictionary from components of the network to the corresponding
            activation.

        Raises:
            ValueError: If the target output_stride is not valid.
        """
        global_pool = self.global_pool
        output_stride = self.output_stride
        reuse = self.resue
        include_root_block = self.include_root_block

        with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.name + '_end_points'
            with arg_scope([layers.conv2d, self.bottleneck,
                            net_resnet_utils.stack_blocks_dense],
                           outputs_collections=end_points_collection):
                with arg_scope([layers.batch_norm], is_training=is_training):
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
                        with arg_scope([layers.conv2d],
                                       activation_fn=None, normalizer_fn=None):
                            net = net_resnet_utils.conv2d_same(
                                net, 64, 7, stride=2, scope='conv1')
                        net = layers.max_pool2d(
                            net, [3, 3], stride=2, scope='pool1')
                    net = net_resnet_utils.stack_blocks_dense(
                        net, blocks, output_stride)
                    # This is needed because the pre-activation variant does not have batch
                    # normalization or activation functions in the residual unit output. See
                    # Appendix of [2].
                    net = layers.batch_norm(
                        net, activation_fn=tf.nn.relu, scope='postnorm')
                    if global_pool:
                        # Global average pooling.
                        net = tf.reduce_mean(
                            net, [1, 2], name='pool5', keep_dims=True)
                    if num_classes is not None:
                        net = layers.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                            normalizer_fn=None, scope='logits')
                    # Convert end_points_collection into a dictionary of
                    # end_points.
                    end_points = utils.convert_collection_to_dict(
                        end_points_collection)
                    if num_classes is not None:
                        end_points['predictions'] = layers.softmax(
                            net, scope='predictions')
                    return net, end_points

    @add_arg_scope
    def bottleneck(self, inputs, depth, depth_bottleneck, stride, rate=1,
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
            depth_in = utils.last_dimension(
                inputs.get_shape(), min_rank=4)
            preact = layers.batch_norm(
                inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = net_resnet_utils.subsample(
                    inputs, stride, 'shortcut')
            else:
                shortcut = layers.conv2d(preact, depth, [1, 1], stride=stride,
                                         normalizer_fn=None, activation_fn=None,
                                         scope='shortcut')

            residual = layers.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                                     scope='conv1')
            residual = net_resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                                    rate=rate, scope='conv2')
            residual = layers.conv2d(residual, depth, [1, 1], stride=1,
                                     normalizer_fn=None, activation_fn=None,
                                     scope='conv3')

            output = shortcut + residual

            return utils.collect_named_outputs(outputs_collections,
                                               sc.original_name_scope,
                                               output)

    def arg_scope(self):
        """Defines the default ResNet arg scope.

        TODO(gpapan): The batch-normalization related default values above are
            appropriate for use in conjunction with the reference ResNet models
            released at https://github.com/KaimingHe/deep-residual-networks. When
            training ResNets from scratch, they might need to be tuned.

        Args:
            weight_decay: The weight decay to use for regularizing the model.
            batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
            batch_norm_epsilon: Small constant to prevent division by zero when
            normalizing activations by their variance in batch normalization.
            batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
            activations in the batch normalization layer.

        Returns:
            An `arg_scope` to use for the resnet models.
        """
        weight_decay = self.weight_decay
        batch_norm_decay = self.batch_norm_decay
        batch_norm_epsilon = self.batch_norm_epsilon
        batch_norm_scale = self.batch_norm_scale

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with arg_scope([layers.conv2d],
                       weights_regularizer=layers.l2_regularizer(weight_decay),
                       weights_initializer=layers.variance_scaling_initializer(),
                       activation_fn=tf.nn.relu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params):
            with arg_scope([layers.batch_norm], **batch_norm_params):
                with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc


class resnet_50(resent):

    def model(self, images, num_classes, is_training):
        """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            net_resnet_utils.Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            net_resnet_utils.Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            net_resnet_utils.Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
            net_resnet_utils.Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(images, blocks, num_classes,
                              is_training=is_training, scope='resnet_v2_50')


class resnet_101(resent):

    def model(self, images, num_classes, is_training):
        """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            net_resnet_utils.Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            net_resnet_utils.Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
            net_resnet_utils.Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
            net_resnet_utils.Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(images, blocks, num_classes,
                              is_training=is_training, scope='resnet_v2_101')


class resnet_152(resent):

    def model(self, images, num_classes, is_training):
        """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            net_resnet_utils.Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            net_resnet_utils.Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
            net_resnet_utils.Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            net_resnet_utils.Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(images, blocks, num_classes,
                              is_training=is_training, scope='resnet_v2_152')


class resnet_200(resent):

    def model(self, images, num_classes, is_training):
        """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
        blocks = [
            net_resnet_utils.Block(
                'block1', self.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
            net_resnet_utils.Block(
                'block2', self.bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
            net_resnet_utils.Block(
                'block3', self.bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
            net_resnet_utils.Block(
                'block4', self.bottleneck, [(2048, 512, 1)] * 3)]
        return self.resnet_v2(images, blocks, num_classes,
                              is_training=is_training, scope='resnet_v2_200')
