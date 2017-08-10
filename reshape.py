import numpy as np
import tensorflow as tf


if __name__ == "__main__":
    array = np.arange(60).reshape(-1, 3, 4)
    aR = tf.reshape(array, [-1, 6])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("=====================================")
        print("From shape: ")
        print(array.shape)
        print(array)
        print("=====================================")
        print("To shape: " + "[-1, 6]")
        print(sess.run(aR))
        print("=====================================")

