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

from preprocessing import cifarnet_preprocessing
from preprocessing import inception_preprocessing
from preprocessing import lenet_preprocessing
from preprocessing import vgg_preprocessing

preprocessing_map = {
    'cifarnet': cifarnet_preprocessing,
    'inception': inception_preprocessing,
    'lenet': lenet_preprocessing,
    'resnet': vgg_preprocessing,
    'vgg': vgg_preprocessing,
}


def check_preprocessing(name, data_type):
    if name not in preprocessing_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)
    if data_type == 'train':
        return True
    elif data_type == 'test':
        return False


def get_preprocessing(name, data_type, image, output_height, output_width, **kwargs):
    is_training = check_preprocessing(name, data_type)
    return preprocessing_map[name].preprocess_image(
        image, output_height, output_width, is_training=is_training, **kwargs)
