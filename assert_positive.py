import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    value_positive = np.arange(1, 10)
    value_negative = np.arange(-10, -1)
    value_check    = np.arange(100, 110)

    print(value_positive)
    print(value_negative)
    print(value_check)
    value_positive = tf.convert_to_tensor(value_positive, dtype=tf.int32) + 1
    value_negative = tf.convert_to_tensor(value_negative, dtype=tf.int32) - 1
    value_check    = tf.convert_to_tensor(value_check, dtype=tf.int32)

    with tf.control_dependencies([tf.assert_positive(value_negative, message="This is Assert Message:============================================")]):
        value_check = tf.identity(value_check)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(value_check))

