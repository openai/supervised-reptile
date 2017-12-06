"""
Train a model on Omniglot.
"""

import tensorflow as tf

from supervised_reptile.args import argument_parser, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/omniglot'
SAVE_DIR = 'omniglot_out'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)

    model = OmniglotModel(args.classes)

    with tf.Session() as sess:
        print('Training...')
        train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
