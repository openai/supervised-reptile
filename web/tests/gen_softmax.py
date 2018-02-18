"""
Generate tests for softmax layers.
"""

import tensorflow as tf

from gen_conv2d import _js_array

# pylint: disable=E1129

def softmax_tests():
    """
    Generate data for softmax tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([5, 3])
        outputs = tf.nn.log_softmax(inputs)
        upstream = tf.random_normal([5, 3])
        grad = tf.gradients(tf.reduce_sum(upstream * outputs), inputs)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run([inputs, outputs, upstream] + grad)
            inputs, outputs, upstream, in_grad = result
            print('const inputs = %s;' % _js_array(inputs))
            print('const outputs = %s;' % _js_array(outputs))
            print('const upstream = %s;' % _js_array(upstream))
            print('const inputGrad = %s;' % _js_array(in_grad))

if __name__ == '__main__':
    softmax_tests()
