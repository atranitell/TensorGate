
import tensorflow as tf

tf.

with tf.device('/gpu:0'):
    p = tf.constant(0)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    p = sess.run (p)