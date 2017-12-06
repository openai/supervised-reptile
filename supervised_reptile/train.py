"""
Training helpers for supervised meta-learning.
"""

import os

import tensorflow as tf

from .reptile import Reptile

# pylint: disable=R0913,R0914
def train(model,
          train_set,
          test_set,
          save_dir,
          num_outer_iters=70000,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          meta_step_size=0.1,
          meta_batch_size=1,
          eval_inner_batch_size=5,
          eval_inner_iters=50):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        reptile = Reptile(sess)
        accuracy_ph = tf.placeholder(tf.float32, shape=())
        tf.summary.scalar('accuracy', accuracy_ph)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
        tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        for i in range(num_outer_iters):
            print('batch %d' % i)
            reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                               num_classes=num_classes, num_shots=num_shots,
                               inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                               meta_step_size=meta_step_size, meta_batch_size=meta_batch_size)
            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters)
                summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes})
                writer.add_summary(summary, i)
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'))
