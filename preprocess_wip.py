import argparse
import csv
import unicodedata
import os
import tensorflow as tf
from random import shuffle

def load_framedata(path="./data/frame-genre-map.txt", drop_rate=0, train_ratio=1):
    # TODO: implement drop_rate, test/train split
    # check if the mapping file is available
    try:
        os.stat(path)
    except:
        print("error: frame to genre map file not found; aborting")
        return None

    inputs = []
    labels = []
    # total number of frames ~130k, takes about 30min?
    with open(path) as map:
        # iterate over each frame -> genre mapping and get array rep of jpgs as inputs,
        # genre as labels
        for i, mapping in enumerate(map):
            frame_path, genre = mapping.split('\t')
            inputs.append(load_and_process_image(frame_path))
            labels.append(genre)
            if i % 1000 == 0:
                print("processed %d frames" % i)
        return inputs, labels

def load_and_process_image(file_path):
    new_size = [100, 185] # dunno if this is good
    # new_size = [244, 244] # used for VGG16
    image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize image
    image = tf.image.resize(image, new_size)
    return image

def get_train_test_data(shuffle=False):
    train_fraction = 0.8

    inputs, labels = load_framedata()
    end_idx = floor(len(labels) * train_fraction)
    train_inputs = inputs[0:end_idx]
    train_labels = labels[0:end_idx]
    test_inputs = inputs[end_idx:]
    test_labels = labels[0:end_idx]

    if shuffle:
      train_indxs = [i for i in range(len(train_labels))]
      test_labels = [i for i in range(len(test_labels))]
      shuffle(train_indxs)
      shuffle(test_indxs)
      train_inputs = [train_inputs[i] for i in train_indxs]
      train_labels = [train_labels[i] for i in train_indxs]
      test_inputs = [test_inputs[i] for i in train_indxs]
      test_labels = [test_labels[i] for i in train_indxs]

    return [train_inputs, train_labels] , [test_inputs, test_labels]

train_data, test_data = get_train_test_data()


# image = tf.image.convert_image_dtype(image, tf.float32)
# resize image
# image = tf.image.resize(image, [100, 185], method='nearest')
# enc = tf.image.encode_jpeg(image)
# fname = tf.constant('asdf.jpg')
# fwrite = tf.io.write_file(fname, enc)
# return image
