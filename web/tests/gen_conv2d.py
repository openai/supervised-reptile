"""
Generate tensor variables for conv2d tests.
"""

import tensorflow as tf

# pylint: disable=E1129

def simple_conv_tests():
    """
    Generate data for convolution tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([1, 2, 2, 1])
        outputs = tf.layers.conv2d(inputs, 1, 3, strides=1, padding='same', use_bias=False)
        upstream = tf.random_normal([1, 2, 2, 1])
        print('Simple conv test:')
        print_conv_case(inputs, outputs, upstream)

def unit_strided_conv_tests():
    """
    Generate data for convolution tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([1, 1, 1, 1])
        outputs = tf.layers.conv2d(inputs, 1, 3, strides=2, padding='same', use_bias=False)
        upstream = tf.random_normal([1, 1, 1, 1])
        print('Unit strided conv test:')
        print_conv_case(inputs, outputs, upstream)

def simple_strided_conv_tests():
    """
    Generate data for convolution tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([1, 2, 2, 1])
        outputs = tf.layers.conv2d(inputs, 1, 3, strides=2, padding='same', use_bias=False)
        upstream = tf.random_normal([1, 1, 1, 1])
        print('Simple strided conv test:')
        print_conv_case(inputs, outputs, upstream)

def complex_conv_tests():
    """
    Generate data for convolution tests.
    """
    with tf.Graph().as_default():
        inputs = tf.random_normal([2, 4, 6, 4])
        outputs = tf.layers.conv2d(inputs, 5, 3, strides=2, padding='same', use_bias=False)
        upstream = tf.random_normal([2, 2, 3, 5])
        print('Complex conv test:')
        print_conv_case(inputs, outputs, upstream)

def print_conv_case(inputs, outputs, upstream):
    """
    Print a conv2d test case as JS variables.
    """
    grads = tf.gradients(tf.reduce_sum(upstream * outputs),
                         [inputs] + tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(tf.trainable_variables() + [inputs, outputs, upstream] + grads)
        kernel, inputs, outputs, upstream, input_grad, kernel_grad = results
        print('const kernel = %s;' % _js_array(kernel))
        print('const inputs = %s;' % _js_array(inputs))
        print('const outputs = %s;' % _js_array(outputs))
        print('const upstream = %s;' % _js_array(upstream))
        print('const inputGrad = %s;' % _js_array(input_grad))
        print('const kernelGrad = %s;' % _js_array(kernel_grad))

def _js_array(numpy_arr):
    return '[%s]' % ', '.join('%.6f' % x for x in numpy_arr.flatten())

if __name__ == '__main__':
    simple_conv_tests()
    unit_strided_conv_tests()
    simple_strided_conv_tests()
    complex_conv_tests()
