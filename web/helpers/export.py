"""
Export a TF model for JavaScript.
"""

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs
from supervised_reptile.models import OmniglotModel

DATA_DIR = '../../data/omniglot'

def main():
    """
    Load the model and train on it.
    """
    args = argument_parser().parse_args()
    OmniglotModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        print('var trainedParameters = [')
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))
        for conv_name in ['', '_1', '_2', '_3']:
            names = [x % conv_name for x in ['conv2d%s/kernel:0',
                                             'batch_normalization%s/gamma:0',
                                             'batch_normalization%s/beta:0']]
            for name in names:
                print_var(sess, name)
                print_var(sess, name.replace(':0', '/Adam_1:0'))
        print_var(sess, 'dense/kernel:0')
        print_var(sess, 'dense/kernel/Adam_1:0')
        print_var(sess, 'dense/bias:0')
        print_var(sess, 'dense/bias/Adam_1:0', last=True)
        print('];')

def print_var(sess, name, last=False):
    """
    Print a variable as a jsnet Tensor.
    """
    var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == name][0]
    val = sess.run(var)
    print('    new jsnet.Tensor([%s], [%s])%s' % (','.join(str(x) for x in val.shape),
                                                  ','.join(str(x) % x for x in val.flatten()),
                                                  '' if last else ','))

if __name__ == '__main__':
    main()

# 'conv2d/kernel:0'
# 'conv2d/bias:0'
# 'batch_normalization/gamma:0'
# 'batch_normalization/beta:0'
# 'batch_normalization/moving_mean:0'
# 'batch_normalization/moving_variance:0'
# 'conv2d_1/kernel:0'
# 'conv2d_1/bias:0'
# 'batch_normalization_1/gamma:0'
# 'batch_normalization_1/beta:0'
# 'batch_normalization_1/moving_mean:0'
# 'batch_normalization_1/moving_variance:0'
# 'conv2d_2/kernel:0'
# 'conv2d_2/bias:0'
# 'batch_normalization_2/gamma:0'
# 'batch_normalization_2/beta:0'
# 'batch_normalization_2/moving_mean:0'
# 'batch_normalization_2/moving_variance:0'
# 'conv2d_3/kernel:0'
# 'conv2d_3/bias:0'
# 'batch_normalization_3/gamma:0'
# 'batch_normalization_3/beta:0'
# 'batch_normalization_3/moving_mean:0'
# 'batch_normalization_3/moving_variance:0'

# 'dense/kernel:0'
# 'dense/bias:0'

# 'beta1_power:0'
# 'beta2_power:0'
# 'conv2d/kernel/Adam:0'
# 'conv2d/kernel/Adam_1:0'
# 'conv2d/bias/Adam:0'
# 'conv2d/bias/Adam_1:0'
# 'batch_normalization/gamma/Adam:0'
# 'batch_normalization/gamma/Adam_1:0'
# 'batch_normalization/beta/Adam:0'
# 'batch_normalization/beta/Adam_1:0'
# 'conv2d_1/kernel/Adam:0'
# 'conv2d_1/kernel/Adam_1:0'
# 'conv2d_1/bias/Adam:0'
# 'conv2d_1/bias/Adam_1:0'
# 'batch_normalization_1/gamma/Adam:0'
# 'batch_normalization_1/gamma/Adam_1:0'
# 'batch_normalization_1/beta/Adam:0'
# 'batch_normalization_1/beta/Adam_1:0'
# 'conv2d_2/kernel/Adam:0'
# 'conv2d_2/kernel/Adam_1:0'
# 'conv2d_2/bias/Adam:0'
# 'conv2d_2/bias/Adam_1:0'
# 'batch_normalization_2/gamma/Adam:0'
# 'batch_normalization_2/gamma/Adam_1:0'
# 'batch_normalization_2/beta/Adam:0'
# 'batch_normalization_2/beta/Adam_1:0'
# 'conv2d_3/kernel/Adam:0'
# 'conv2d_3/kernel/Adam_1:0'
# 'conv2d_3/bias/Adam:0'
# 'conv2d_3/bias/Adam_1:0'
# 'batch_normalization_3/gamma/Adam:0'
# 'batch_normalization_3/gamma/Adam_1:0'
# 'batch_normalization_3/beta/Adam:0'
# 'batch_normalization_3/beta/Adam_1:0'
# 'dense/kernel/Adam:0'
# 'dense/kernel/Adam_1:0'
# 'dense/bias/Adam:0'
# 'dense/bias/Adam_1:0'
