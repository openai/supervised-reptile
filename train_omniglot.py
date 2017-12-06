"""
Train a model on Omniglot.
"""

from supervised_reptile.models import OmniglotModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = 'data/omniglot'
SAVE_DIR = 'omniglot_out'

def main():
    """
    Load data and train a model on it.
    """
    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = augment_dataset(train_set)
    train(OmniglotModel(5), list(train_set), list(test_set), SAVE_DIR)

if __name__ == '__main__':
    main()
