import argparse
import csv
import unicodedata
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

"""
Casts inputs as filepaths to images, and labels as onehotencoded arrays
Splits the inputs and labels into a train test split.
returns X_train, X_test, y_train, y_test

OneHotEncoding needs to be adjusted for multiclass
"""
# dropout in model
def load_framedata(path="./data/frame-genre-map.txt", train_ratio=1):
    # TODO: implement drop_rate, test/train split
    # check if the mapping file is available
    try:
        os.stat(path)
    except:
        print("error: frame to genre map file not found; aborting")
        return None

    inputs = []
    labels = []
    with open(path) as map:
        # genre as labels
        for i, mapping in enumerate(map):
            frame_path, genres = mapping.split('\t')
            print('hi')
            print(genres)
            genres = genres.strip()
            genres = genres.split(',')

            inputs.append(frame_path)
            labels.append([genres[0]])
            labels.append([genres[1]])
            print(labels) # just getting the first genre for now

        # onehotencoding labels, needs to be editted for multiclass
    enc = OneHotEncoder()
    enc.fit(labels)
    labels = enc.transform(labels).toarray()

    # SKlearn train_test_split. Automatically shuffles the data
    return train_test_split(inputs, labels, test_size=0.2, random_state=42)
