import argparse
import csv
import unicodedata
import os
import tensorflow as tf
import numpy as np

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
    new_size = [128, 176] # dunno if this is good

    image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # resize image
    image = tf.image.resize(image, new_size)
    return image

inputs, labels = load_framedata()
print('converting to numpy...')
inputs = np.array(inputs)
labels = np.array(labels)
print('saving....')
np.save('inputs', inputs)
np.save('labels', labels)


# image = tf.image.convert_image_dtype(image, tf.float32)
# resize image
# image = tf.image.resize(image, [100, 185], method='nearest')
# enc = tf.image.encode_jpeg(image)
# fname = tf.constant('asdf.jpg')
# fwrite = tf.io.write_file(fname, enc)
# return image
    
    

    
