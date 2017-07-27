import numpy as np
import tensorflow as tf

# tf.unstack input like:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
#
# tf.unstack axis=0 output is:
# [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([ 8,  9, 10, 11])]
#
# tf.unstack axis=1 output is
# [array([0, 4, 8]), array([1, 5, 9]), array([ 2,  6, 10]), array([ 3,  7, 11])]

if __name__ == "__main__":
    dim = 8

    raw_data = np.arange(12).reshape([3, 4])
    unpack_H = tf.unstack(raw_data, axis=0)
    unpack_V = tf.unstack(raw_data, axis=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(raw_data)
        print(sess.run(unpack_H))
        print(sess.run(unpack_V))
