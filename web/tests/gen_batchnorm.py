"""
Generate tests for batch normalization.
"""

import tensorflow as tf

from gen_conv2d import _js_array

# pylint: disable=E1129

def batchnorm_tests():
    """
    Generate data for batch norm tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([5, 3])
        outputs = tf.contrib.layers.batch_norm(inputs, is_training=True)
        upstream = tf.random_normal([5, 3])
        grad = tf.gradients(tf.reduce_sum(upstream * outputs), [inputs] + tf.trainable_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.assign(tf.trainable_variables()[0], tf.random_normal([3])))
            result = sess.run([inputs, outputs, upstream] + grad + tf.trainable_variables())
            inputs, outputs, upstream, in_grad, var_grad, beta = result
            print('const inputs = %s;' % _js_array(inputs))
            print('const outputs = %s;' % _js_array(outputs))
            print('const upstream = %s;' % _js_array(upstream))
            print('const inputGrad = %s;' % _js_array(in_grad))
            print('const betaGrad = %s;' % _js_array(var_grad))
            print('const beta = %s;' % _js_array(beta))

if __name__ == '__main__':
    batchnorm_tests()
