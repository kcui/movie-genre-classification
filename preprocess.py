import argparse
import csv
import unicodedata
import os
import tensorflow as tf
import numpy as nps
import math
import random
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
            print("hey")
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
        for mapping in map:
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

        # ~80% at 93977
        X_train, y_train, X_test, y_test = inputs[:93977], labels[:93977], inputs[93977:95000], labels[93977:95000]
        
        # X_train, y_train = shuffle(X_train, y_train)

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        return X_train, X_test, y_train, y_test, enc


"""
A normalized split of the database, where each valid genre is assigned an equal number of movies.
If test_size is specified, no movie in the train set will appear in the test set and vice versa
"""
def split_on_movie_normalized(path="./data/frame-genre-map.txt", movies_per_genre=60, test_size=0.2):
    try:
        os.stat(path)
    except:
        print("error: frame to genre map file not found; aborting")
        return None

    # valid_genres = ['Drama', 'Adventure', 'Crime', 'Action', 'Biography', 
    # 'Comedy', 'Horror', 'Mystery', 'Romance', 'Fantasy', 'Thriller'],

    valid_genres = ['Action','Comedy', 'Horror', 'Romance']

    genre_to_movie = {}

    with open(path) as map:
        for mapping in map:
            frame_path, genres = mapping.split('\t')

            genres = genres.strip()
            genres = genres.split(',')

            mov = frame_path.split('/')[2]

            for genre in genres:
                if genre not in valid_genres:
                    continue
                if genre not in genre_to_movie:
                    genre_to_movie[genre] = set()
                genre_to_movie[genre].add(mov)

    for genre in genre_to_movie.keys():
        genre_to_movie[genre] = random.sample(genre_to_movie[genre], movies_per_genre)

    inputs = []
    labels = []

    if test_size:
        split_amt = math.ceil(movies_per_genre * (1 - test_size))
        test_inputs = []
        test_labels = []
        train_movs_per_genre = {}

        for genre in valid_genres:
            train_movs_per_genre[genre] = set()

    with open(path) as map:
        # genre as labels
        for mapping in map:
            frame_path, genres = mapping.split('\t')
            genres = genres.strip()
            genres = genres.split(',')

            mov = frame_path.split('/')[2]
            
            for genre in genres:
                if genre not in genre_to_movie:
                    continue
                if mov in genre_to_movie[genre]:
                    if test_size and len(train_movs_per_genre[genre]) >= split_amt and mov not in train_movs_per_genre[genre]:
                        test_inputs.append(frame_path)
                        test_labels.append([genre])
                    else:
                        inputs.append(frame_path)
                        labels.append([genre]) #single class

                        if test_size:
                            train_movs_per_genre[genre].add(mov)
                    break

        enc = OneHotEncoder()
        
        if test_size:
            enc.fit(labels + test_labels)
            labels = enc.transform(labels).toarray()
            test_labels = enc.transform(test_labels).toarray()
            X_train, y_train = shuffle(inputs, labels)
            X_test, y_test = shuffle(test_inputs, test_labels)
        else:
            enc.fit(labels)
            labels = enc.transform(labels).toarray()
            X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

        # test_mov_set = set()
        # for i in X_test:
        #     test_mov_set.add(i.split('/')[2])
        # train_mov_set = set()
        # for i in X_train:
        #     train_mov_set.add(i.split('/')[2])
        
        # SKlearn train_test_split. Automatically shuffles the data
        return X_train, X_test, y_train, y_test, enc