"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for miniImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)

# pylint: disable=R0903
class ResNetMiniImageNetModel:
    """
    A model for miniImageNet classification based on
    a simplified ResNet-18.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 88, 88, 3))
        out = self.input_ph
        def _single_block(inputs, num_features, stride=1):
            res = tf.layers.conv2d(inputs, num_features, 3, strides=stride, padding='same')
            res = tf.layers.batch_normalization(res, training=True)
            res = tf.nn.relu(res)
            res = tf.layers.conv2d(res, num_features, 3, padding='same')
            return tf.layers.batch_normalization(res, training=True)
        out = tf.layers.conv2d(out, 32, 3, padding='same')
        for i, num_features in enumerate([32, 64]):
            if i > 0:
                # Project residual connections.
                residual_value = tf.layers.conv2d(out, num_features, 1, strides=2, padding='same')
                residual_value = tf.layers.batch_normalization(residual_value)
                out = _single_block(out, num_features, stride=2)
            else:
                residual_value = out
                out = _single_block(out, num_features)
            out = tf.nn.relu(out + residual_value)
            out = tf.nn.relu(out + _single_block(out, num_features))
        out = tf.reduce_mean(out, axis=[1, 2])
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
