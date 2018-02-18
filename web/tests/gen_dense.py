"""
Generate tests for dense layers.
"""

import tensorflow as tf

from gen_conv2d import _js_array

# pylint: disable=E1129

def dense_tests():
    """
    Generate data for dense tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([5, 3])
        outputs = tf.layers.dense(inputs, 4, use_bias=False)
        upstream = tf.random_normal([5, 4])
        grad = tf.gradients(tf.reduce_sum(upstream * outputs), [inputs] + tf.trainable_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run([inputs, outputs, upstream] + grad + tf.trainable_variables())
            inputs, outputs, upstream, in_grad, weights_grad, weights = result
            print('const inputs = %s;' % _js_array(inputs))
            print('const outputs = %s;' % _js_array(outputs))
            print('const upstream = %s;' % _js_array(upstream))
            print('const inputGrad = %s;' % _js_array(in_grad))
            print('const weightsGrad = %s;' % _js_array(weights_grad))
            print('const weights = %s;' % _js_array(weights))

if __name__ == '__main__':
    dense_tests()
