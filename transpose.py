import numpy as np
import tensorflow as tf


#
# tf.transpose
# exchange dimentions, eg: 2, 3, 4 change to 3, 2, 4
#

dim1_size = 2
dim2_size = 3
dim3_size = 4

dim1 = 0
dim2 = 1
dim3 = 2

if __name__ == "__main__":
    raw = np.arange(dim1_size*dim2_size*dim3_size)
    raw = raw.reshape(dim1_size, dim2_size, dim3_size)

    data = tf.transpose(raw, [dim2,dim1,dim3])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("================================")
        print(raw)
        print(raw.shape)
        print("================================")
        print(sess.run(data))
        print(data)
        print("================================")
