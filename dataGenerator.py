from skimage.io import imread
from skimage.transform import resize
import numpy as np
import math
from keras.utils import Sequence

"""
Class for generating data on the dataset

Takes in file paths, batches them, and returns images and their respective labels.
"""
class dataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (128, 176, 3))
               for file_name in batch_x], dtype=np.float32), np.array(batch_y)