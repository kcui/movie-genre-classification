import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from preprocess import load_framedata, split_on_movie
from dataGenerator import dataGenerator


def setup_model(shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(shape[0], shape[1], 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='sigmoid'))

    adam = Adam() # tune adam parameters possibly
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

"""
Returns the genre class for a movie, depending on the most frequently classified frame genres.

returns a map from movie to the predicted genre for that movie (onehot)
"""
def movie_preds(inputs, frame_preds):

    mov_dict = {}

    for i in range(len(inputs)):
        path = inputs[i]
        pred = frame_preds[i]

        mov = path.split('/')[2]
        if mov not in mov_dict:
            mov_dict[mov] = {}

        if pred not in mov_dict[mov]:
            mov_dict[mov][pred] = 1
        else:
            mov_dict[mov][pred] += 1

    mov_to_label = {}

    # currently assigns the most frequently predicted genre to the movie
    for mov in mov_dict.keys():
        max_pred = None
        max_pred_val = 0

        for pred in mov_dict[mov].keys():
            if mov_dict[mov][pred] > max_pred_val:
                max_pred = pred
                max_pred_val = mov_dict[mov][pred]

        mov_to_label[mov] = max_pred

    return mov_to_label

"""
Returns the fraction of correctly labeled movies
"""
def test_accuracy(pred_dict, label_dict):

    correct = 0

    for mov in pred_dict.keys():
        if pred_dict[mov] == label_dict[mov]:
            correct += 1

    return correct / len(pred_dict)

"""
Converts genre categories from onehot labels back to genre strings
"""
def convert_onehot_to_genre(pred_dict, label_dict, num_to_genre):

    for mov in pred_dict.keys():
        pred_dict[mov] = num_to_genre[pred_dict[mov]]
        label_dict[mov] = num_to_genre[label_dict[mov]]

    return pred_dict, label_dict

def run_model(multiclass=True):
    # X_train, X_test, y_train, y_test, encoder = load_framedata() # loads in data from preprocess
    X_train, X_test, y_train, y_test, encoder = split_on_movie()

    # Used to convert from onehot labels back to genre strings

    if multiclass:
        cats = encoder.classes_
        print(cats)
    else:
        cats = encoder.categories_[0]
    num_to_genre = {}
    for i, genre in enumerate(cats):
        num_to_genre[i] = genre

    train_generator = dataGenerator(X_train, y_train, batch_size=32) # see datagenerator class
    test_generator = dataGenerator(X_test, y_test, batch_size=32)

    model = setup_model((128, 176), num_classes=28)

    try:
        os.stat('/model_1.h5')
        print("loading model..")
        model = load_model('./model_1.h5')
    except:
        print("no preloaded model. training model...")
        model.fit_generator(train_generator, epochs=1, verbose=1)
        print("saving model...")
        model.save('model_1.h5')

    print('------- Testing model -------')

    # score = model.evaluate_generator(test_generator)
    # print("Test Metrics: ", score)

    frame_preds = model.predict_generator(test_generator) # generate prediction labels on the frames
<<<<<<< HEAD
    
    pred_dict = movie_preds(X_test, frame_preds.flatten()) # movie -> genre (onehot)
=======

    pred_dict = movie_preds(X_test, frame_preds) # movie -> genre (onehot)
>>>>>>> c8fe6d0dad45c8094c31cbfd76f027b7ba2af2dc
    label_dict = movie_preds(X_test, y_test) # movie -> genre (onehot)

    print('Test Accuracy predicting Movie Genres: ', test_accuracy(pred_dict, label_dict)) # accuracy

    # pred_dict, label_dict = convert_onehot_to_genre(pred_dict, label_dict, num_to_genre)

    # print('Movie\tPredicted\tActual')

    # for mov in pred_dict.keys():
    #     print("%s\t%s\t%s", mov, pred_dict[mov], label_dict[mov])


if __name__ == "__main__":
    # setup_model()
    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)
    run_model()
