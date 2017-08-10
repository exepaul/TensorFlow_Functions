import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    array1_dim = 4
    array2_dim = 15
    array3_dim = 11
    raw_data = np.arange(150).reshape([-1, array1_dim + array2_dim + array3_dim])
    print raw_data
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        split0,  split1, split2 = tf.split(raw_data, [array1_dim, array2_dim, array3_dim], axis=1)
        print("=====================================================")
        print(sess.run(split0))
        print(sess.run(split1))
        print(sess.run(split2))

        print("=====================================================")
        split0,  split1, split2 = tf.split(raw_data, num_or_size_splits=3, axis=1)
        print(sess.run(split0))
        print(sess.run(split1))
        print(sess.run(split2))
        print("=====================================================")
