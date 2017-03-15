
import tensorflow as tf


a = tf.constant(['a1', 'a2', 'a3', 'a4'])
b = tf.constant(['k1', '\r\n', 'k3', 'k4'])
c = tf.constant([1,2,3,4])
c = tf.as_string(c)
with tf.Session() as sess:
    print(sess.run(a+b))
    print(sess.run(a+b+c))