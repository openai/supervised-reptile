"""
Loading and using the miniImageNet dataset.
"""

import os
import random

from PIL import Image
import numpy as np

# pylint: disable=R0903
class ImageNetClass:
    """
    A single image class.
    """
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 84x84x3 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.jpg')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            with open(os.path.join(self.dir_path, name), 'rb') as in_file:
                img = Image.open(in_file).resize((84, 84))
                images.append(np.array(img).astype('float32') / 0xff)
        return images
