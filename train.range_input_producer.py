import numpy as np
import codecs
import tensorflow as tf

BATCH_SIZE   = 6
NUM_EXPOCHES = 5

def input_producer():
    array = np.arange(100)
    i = tf.train.range_input_producer(NUM_EXPOCHES, num_epochs=1, shuffle=False).dequeue()

    inputs = tf.slice(array, [i * BATCH_SIZE], [BATCH_SIZE])
    return inputs

class Inputs(object):
    def __init__(self):
        self.inputs = input_producer()


def main(*args, **kwargs):
    inputs = Inputs()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init)
        
        try:
            index = 0
            while not coord.should_stop() and index < 5:
                dataline = sess.run(inputs.inputs)
                index += 1
                print("steps: %d, batch data: %s" % (index, str(dataline)))
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
