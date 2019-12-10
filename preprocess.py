import argparse
import csv
import unicodedata
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
Casts inputs as filepaths to images, and labels as onehotencoded arrays
Splits the inputs and labels into a train test split.
returns X_train, X_test, y_train, y_test

OneHotEncoding needs to be adjusted for multiclass
"""
# dropout in model
def load_framedata(multiclass, path="./data/frame-genre-map.txt", train_ratio=1):
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
            genres = genres.strip()
            genres = genres.split(',')

            inputs.append(frame_path)
            if multiclass:
                labels.append(genres) # multiclass
            else:
                labels.append([genres[0]]) #single class

        if multiclass:
            enc = MultiLabelBinarizer()
            labels = enc.fit_transform(labels)
        else:
            enc = OneHotEncoder()
            enc.fit(labels)
            labels = enc.transform(labels).toarray()

        X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

        # SKlearn train_test_split. Automatically shuffles the data
        return X_train, X_test, y_train, y_test, enc

def split_on_movie(path="./data/frame-genre-map.txt", multiclass=True):
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
            genres = genres.strip()
            genres = genres.split(',')

            inputs.append(frame_path)
            if multiclass:
                labels.append(genres) # multiclass
            else:
                labels.append([genres[0]]) #single class

        if multiclass:
            enc = MultiLabelBinarizer()
            labels = enc.fit_transform(labels)
        else:
            enc = OneHotEncoder()
            enc.fit(labels)
            labels = enc.transform(labels).toarray()

        inputs, labels = shuffle(inputs, labels)
        # ~80% at 93977
        X_train, y_train, X_test, y_test = inputs[:93977], labels[:93977], inputs[93977:], labels[93977:]

        #X_train, y_train = shuffle(X_train, y_train)
        #X_test, y_test = shuffle(X_test, y_test)

        return X_train, X_test, y_train, y_test, enc
