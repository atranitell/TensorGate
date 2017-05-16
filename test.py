import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import tensorflow as tf

a = tf.constant([[1., 2, 3]])
b = tf.constant([[-1., -2, -3]])

with tf.Session() as sess:
    x = sess.run(tf.matmul(a/tf.norm(a), b/tf.norm(b), transpose_b=True))
    print(x)