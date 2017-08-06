import numpy as np
import tensorflow as tf

# ========================================
#  http://blog.csdn.net/banana1006034246/article/details/75092388
#  ========================================

data = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11,12,13,14,15,16,17,18]
     ]



if __name__ == "__main__":
    x_dim1_start = 0
    x_dim1_end  = 1
    x_dim2_start = 0
    x_dim2_end  = 4
    x = tf.strided_slice(data, [x_dim1_start, x_dim2_start], [x_dim1_end, x_dim2_end])
    y = tf.strided_slice(data, [1, 1], [2, 4])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(data)
        # x output shoud be [1, 2, 3, 4]  dim1=0, dim2=0~4
        print(sess.run(x))
        print(sess.run(y))

