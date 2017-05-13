"""Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
      ValueError: If Preprocessing `name` is not recognized.
"""

import tensorflow as tf

from gate.preprocessing import preprocessing_cifarnet
from gate.preprocessing import preprocessing_inception
from gate.preprocessing import preprocessing_lenet
from gate.preprocessing import preprocessing_vgg
from gate.preprocessing import preprocessing_mnist
from gate.preprocessing import preprocessing_mnist_gan
from gate.preprocessing import preprocessing_avec2013
from gate.preprocessing import preprocessing_avec2014
from gate.preprocessing import preprocessing_avec2014_flow
from gate.preprocessing import preprocessing_celeba_gan
from gate.preprocessing import preprocessing_kinface


preprocessing_map = {
    'cifarnet': preprocessing_cifarnet,
    'inception': preprocessing_inception,
    'lenet': preprocessing_lenet,
    'resnet': preprocessing_vgg,
    'vgg': preprocessing_vgg,
    'mnist': preprocessing_mnist,
    'mnist_gan': preprocessing_mnist_gan,
    'avec2014': preprocessing_avec2014,
    'avec2013': preprocessing_avec2013,
    'avec2014_flow': preprocessing_avec2014_flow,
    'celeba_gan': preprocessing_celeba_gan,
    'kinface': preprocessing_kinface
}


def check_preprocessing(name, data_type):
    """
    """
    if name not in preprocessing_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)
    if data_type == 'train':
        return True
    elif data_type == 'test':
        return False


def get_preprocessing(name, data_type, image, output_height, output_width, **kwargs):
    """
    """
    is_training = check_preprocessing(name, data_type)
    with tf.variable_scope('preprocessing'):
        return preprocessing_map[name].preprocess_image(
            image, output_height, output_width, is_training=is_training, **kwargs)
