import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

import tensorflow as tf


x = tf.constant([1, 1, 1], dtype=tf.float32)
y = tf.constant([2, 2, 2], dtype=tf.float32)

z = tf.reduce_mean([x, y], axis=0)

z1 = tf.concat(values=[x, y, z], axis=0)

with tf.Session() as sess:
    print(sess.run(z1))