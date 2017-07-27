import numpy as np
import tensorflow as tf


# tf.one_hot imput like::
# [1 1 4 4 2]
#
# tf.one_hot output is:
# [[ 0.  1.  0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  1.  0.  0.  0.]
#  [ 0.  0.  0.  0.  1.  0.  0.  0.]
#  [ 0.  0.  1.  0.  0.  0.  0.  0.]]

if __name__ == "__main__":
    dim = 8

    raw_data = np.random.choice(dim, size=[5, ])
    one_hot = tf.one_hot(raw_data, dim)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(raw_data)
        print(sess.run(one_hot))
