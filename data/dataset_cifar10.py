# -*- coding: utf-8 -*-
# ./data/datasets_data_provider.py
#
#    Tensorflow Version: r1.0
#    Python Version: 3.5
#    Update Date: 2017/03/13
#    Author: Kai JIN
# ==============================================================================

from data import dataset
from data import utils
import tensorflow as tf


class cifar10(dataset.Dataset):

    def __init__(self, data_type):
        self.data_type = data_type

        if data_type == 'train':
            self._init_train_param()
        elif data_type == 'test':
            self._init_test_param()
        else:
            raise ValueError('Unknown command %s' % self.data_type)

        self._init_common_param()
        self._init_lrn_param()
        self._init_opt_param()
        self._init_log_param()

    def _init_log_param(self):
        class log_param():
            pass
        self.log = log_param()
        # Directory where checkpoints and event logs are written to.
        if self.data_type == 'train':
            self.log.train_dir = utils.dir_log_constructor(
                '_output/cifar_train')
        elif self.data_type == 'test':
            self.log.test_dir = None
        # The frequency with which logs are print.
        self.log.print_frequency = 50
        # The frequency with which summaries are saved, in iteration.
        self.log.save_summaries_iter = 500
        # The frequency with which the model is saved, in iteration.
        self.log.save_model_iter = 1000
        # test iteration
        self.log.test_interval = 1000

    def _init_opt_param(self):
        """The name of the optimizer: 
        Args: 
            "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd", "rmsprops"
        """
        class opt_param():
            pass
        self.opt = opt_param()

        self.opt.optimizer = 'adagrad'

        """ SGD """
        self.opt.weight_decay = 0.00004
        self.opt.momentum = 0.9
        self.opt.opt_epsilon = 1.0

        """ ADAGRAD """
        self.opt.adadelta_rho = 0.95
        self.opt.adagrad_initial_accumulator_value = 0.1

        """ ADAMs """
        self.opt.adam_beta1 = 0.9
        self.opt.adam_beta2 = 0.999

        """ FTRL """
        self.opt.ftrl_learning_rate_power = -0.5
        self.opt.ftrl_initial_accumulator_value = 0.1
        self.opt.ftrl_l1 = 0.0
        self.opt.ftrl_l2 = 0.0

        """ RMSProp """
        self.opt.rmsprop_momentum = 0.9
        # Decay term for RMSProp.
        self.opt.rmsprop_decay = 0.9

    def _init_lrn_param(self):
        class opt():
            pass
        self.lr = opt()
        # Specifies how the learning rate is decayed. One of "fixed",
        # "exponential", or "polynomial"
        self.lr.learning_rate_decay_type = 'fixed'
        # Initial learning rate.
        self.lr.learning_rate = 0.01
        # The minimal end learning rate used by a polynomial decay learning
        # rate.
        self.lr.end_learning_rate = 0.0001
        # The amount of label smoothing.
        self.lr.label_smoothing = 0.0
        # Learning rate decay factor
        self.lr.learning_rate_decay_factor = 0.94
        # Number of epochs after which learning rate decays.
        self.lr.num_epochs_per_decay = 2.0
        # Whether or not to synchronize the replicas during training.
        self.lr.sync_replicas = False
        # The Number of gradients to collect before updating params.
        self.lr.replicas_to_aggregate = 1
        # The decay to use for the moving average.
        # If left as None, then moving averages are not used.
        self.lr.moving_average_decay = None

    def _init_common_param(self):
        self.batch_size = 32
        self.output_height = 28
        self.output_width = 28
        self.padding = 4
        self.reader_thread = 1
        self.min_queue_num = 1024
        self.device = '/cpu:0'
        self.num_classes = 10
        self.preprocessing_method = 'cifarnet_preprocessing'

    def _init_train_param(self):
        self.total_num = 50000
        self.name = 'cifar10_train'
        self.shuffle = True
        self.data_path = 'C:/Users/jk/Desktop/Tensorflow/mood-new/Video-Mood/_datasets/cifar10/train_list.txt'

    def _init_test_param(self):
        self.total_num = 10000
        self.name = 'cifar10_test'
        self.shuffle = False
        self.data_path = 'C:/Users/jk/Desktop/Tensorflow/mood-new/Video-Mood/_datasets/cifar10/test_list.txt'

    def loads(self):
        """ load images and labels from folder/files.
            1) load in queue 2) preprocessing 3) output batch

        Note:
            There, we will load image from train.list(like caffe)

        Returns:
            images: 4D tensor of [batch_size, height, wideth, channel] size.
            labels: 1D tensor of [batch_size] size.
        """

        file_list_path = self.data_path
        batch_size = self.batch_size
        total_num = self.total_num
        image_list, label_list, load_num = utils.read_from_file(file_list_path)

        if total_num != load_num:
            raise ValueError('Loading in %d images, but setting is %d images!' %
                             (load_num, total_num))

        # construct a fifo queue
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer(
            [images, labels], shuffle=self.shuffle)

        # preprocessing
        # there, the cifar image if 'JPEG' format
        image_raw = tf.read_file(input_queue[0])
        image_jpeg = tf.image.decode_jpeg(image_raw, channels=3)
        image = self._preprocessing_image(
            self.preprocessing_method, self.data_type,
            image_jpeg, self.output_height, self.output_width)

        # preprocessing method
        label = self._preprocessing_label(input_queue[1])

        return self._generate_image_label_batch(
            image, label, self.shuffle, self.min_queue_num,
            self.batch_size, self.reader_thread)

    # def _preprocessing_image(self, image):
    #     if self.data_type is 'train':
    #         return self._preprocessing_image_for_train(image)
    #     elif self.data_type is 'test':
    #         return self._preprocessing_image_for_test(image)

    # def _preprocessing_image_for_train(self, image):
    #     """ Preprocesses the given image for training.

    #     Note that the actual resizing scale is sampled from
    #         [`resize_size_min`, `resize_size_max`].

    #     Args:
    #         image: A `Tensor` representing an image of arbitrary size.
    #         output_height: The height of the image after preprocessing.
    #         output_width: The width of the image after preprocessing.
    # padding: The amound of padding before and after each dimension of the
    # image.

    #     Returns:
    #         A preprocessed image.
    #     """

    #     output_height = self.output_height
    #     output_width = self.output_width
    #     padding = self.padding

    #     tf.summary.image('image', tf.expand_dims(image, 0))

    #     # Transform the image to floats.
    #     image = tf.to_float(image)
    #     if padding > 0:
    #         image = tf.pad(image, [[padding, padding],
    #                                [padding, padding], [0, 0]])
    #     # Randomly crop a [height, width] section of the image.
    #     distorted_image = tf.random_crop(image,
    #                                      [output_height, output_width, 3])

    #     # Randomly flip the image horizontally.
    #     distorted_image = tf.image.random_flip_left_right(distorted_image)

    #     tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

    #     # Because these operations are not commutative, consider randomizing
    #     # the order their operation.
    #     distorted_image = tf.image.random_brightness(
    #         distorted_image, max_delta=63)
    #     distorted_image = tf.image.random_contrast(
    #         distorted_image, lower=0.2, upper=1.8)

    #     # Subtract off the mean and divide by the variance of the pixels.
    #     return tf.image.per_image_standardization(distorted_image)

    # def _preprocessing_image_for_test(self, image):
    #     """ Preprocesses the given image for test. """

    #     output_height = self.output_height
    #     output_width = self.output_width

    #     tf.summary.image('image', tf.expand_dims(image, 0))
    #     # Transform the image to floats.
    #     image = tf.to_float(image)

    #     # Resize and crop if needed.
    #     resized_image = tf.image.resize_image_with_crop_or_pad(
    #         image, output_width, output_height)
    #     tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

    #     # Subtract off the mean and divide by the variance of the pixels.
    #     return tf.image.per_image_standardization(resized_image)

    # def _preprocessing_label(self, label):
    #     if self.data_type is 'train':
    #         return self._preprocessing_label_for_train(label)
    #     elif self.data_type is 'test':
    #         return self._preprocessing_label_for_test(label)

    # def _preprocessing_label_for_train(self, label):
    #     return label

    # def _preprocessing_label_for_test(self, label):
    #     return label


