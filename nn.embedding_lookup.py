import numpy as np
import tensorflow as tf


#
# tf.nn.embedding_lookup(embedding, input_ids)
# embedding 是字典，input_ids通过 索引找到embedding 中对应的数据
#


if __name__ == "__main__":
    dim = 6

    input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
    identity_array = np.identity(dim, dtype=np.int32)
    embedding = tf.Variable(identity_array)
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
    X = np.random.choice(dim, 7)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("embedding is :\n", sess.run(embedding))
        print("input_ids is :\n\t", X)
        print("output is :\n", sess.run([input_embedding], feed_dict={input_ids:X}))