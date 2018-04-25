
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from gate.util.logger import logger
from gate.config.mnist import MNIST
from gate.util.string import print_members
from gate.config.factory import get_config
from gate.data.factory import get_data

logger.init('1', './')
config = get_config('mnist')

config.set_phase(config.phase)


image, label, path = get_data(config)
print(config.data.total_num)
print(image)


# a = MNIST('123')
# print_members(a)
