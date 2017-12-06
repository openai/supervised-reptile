"""
Models for supervised meta-learning.
"""

import tensorflow as tf

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for padding in ['same', 'same', 'valid', 'valid']:
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding=padding)
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = tf.train.AdamOptimizer(beta1=0).minimize(self.loss)
