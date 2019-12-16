import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import pickle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from preprocess import load_framedata, split_on_movie, split_on_movie_normalized
from dataGenerator import dataGenerator
import sklearn
import matplotlib.pyplot as plt

def setup_model_multiclass(shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(shape[0], shape[1], 3)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same',))
    model.add(LeakyReLU())
    # model.add(Conv2D(64, (3, 3), padding='same',))
    # model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), padding='same',))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='sigmoid'))

    adam = Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999) # tune adam parameters possibly
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def setup_model_singleclass(shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(shape[0], shape[1], 3)))
    model.add(LeakyReLU())
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), padding='same',))
    model.add(LeakyReLU())
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same',))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    #
    #model.add(Conv2D(128, (3, 3), padding='same',))
    #model.add(LeakyReLU())
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same',))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.1))
    #
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same',))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.1))
    #model.add(Dense(2048, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam(learning_rate=0.001, beta_1=0.5, beta_2=0.999) # tune adam parameters possibly
    sgd = SGD(lr=0.01, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

"""
Returns the genre class for a movie, depending on the most frequently classified frame genres.

returns a map from movie to the predicted genre for that movie (onehot)
"""
def movie_preds(inputs, frame_preds):

    mov_dict = dict()

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
Compute hamming score for multilabel predictions
"""
def hamming_score(y_pred, y_true):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        curr_acc = None
        if len(set_true) == 0 and len(set_pred) == 0:
            curr_acc = 1
        else:
            curr_acc = len(set_true.intersection(set_pred))/\
                    float(len(set_true.union(set_pred)))
        acc_list.append(curr_acc)
    return np.mean(acc_list)

"""
Return top 3 predicted genres from probabilities as one-hot labels
"""

def probs_to_preds(probs):
    preds = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        curr_probs = probs[i]
        top3 = np.partition(probs[i].flatten(), -3)[-3]
        curr_probs[curr_probs>=top3] = 1
        curr_probs[curr_probs<top3] = 0
        preds[i] = curr_probs
    print(preds[0:3])
    return preds

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

def plot_model(history, fromDict = True):
    # Plot training & validation accuracy values
    if not fromDict:
        history = history.history
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('accuracy.png')

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('loss.png')

def run_model(multiclass=False, normalized=True):
    if normalized and not multiclass: # NEW TEST_SIZE SPECIFIED HERE
        X_train, X_test, y_train, y_test, encoder = split_on_movie_normalized(multiclass=multiclass, test_size=0.2)
        num_classes = 4
    elif normalized and multiclass:
        X_train, X_test, y_train, y_test, encoder = split_on_movie_normalized(multiclass=multiclass, test_size=0.2)
        print(y_train)
        num_classes=8

    # Used to convert from onehot labels back to genre strings
    print(np.sum(y_train, axis=0))
    print(np.sum(y_test, axis=0))
    # print(y_test)
    # print(sum(y_test))

    if multiclass:
        cats = encoder.classes_
        print(cats)
        print(len(cats))
    else:
        cats = encoder.categories_[0]
    num_to_genre = {}
    for i, genre in enumerate(cats):
        num_to_genre[i] = genre

    train_generator = dataGenerator(X_train, y_train, batch_size=32) # see datagenerator class
    validation_generator = dataGenerator(X_test, y_test, batch_size=32)
    test_generator = dataGenerator(X_test, y_test, batch_size=32)

    try:
        filepath = './model_1.h5'
        #filepath = './None'
        os.stat(filepath)
        print("existing model found; loading model...")
        model = load_model(filepath)

        # Plot history from dictionary
        with open('./trainHistoryDict', 'rb') as file_pi:
            history = pickle.load(file_pi)
            plot_model(history, fromDict = True)
    except:
        print("no preloaded model. training model...")
        if multiclass:
            model = setup_model_multiclass((128, 176), num_classes=num_classes)
        else:
            model = setup_model_singleclass((128, 176), num_classes=num_classes)
        history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=5, verbose=1)
        print("saving model...")
        model.save('model_1.h5')
        print("graphing model and saving history...")

        # Save history
        with open('trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        plot_model(history, fromDict = False)

    print('------- Testing model -------')

    frame_preds = model.predict_generator(test_generator, verbose=1) # generate prediction labels on the frames

    if multiclass:
        # categorical accuracy
        acc = tf.keras.metrics.CategoricalAccuracy()
        acc.update_state(frame_preds, y_test)
        result = acc.result()
        print("categorical accuracy: ")
        print(result)

        # hamming score
        preds = probs_to_preds(frame_preds)
        score = hamming_score(preds, y_test)
        print("hamming score: ")
        print(score)

    else:
        true_labels = [np.where(r==1)[0][0] for r in y_test]
        predicted_labels = np.argmax(frame_preds, axis=1)

        pred_dict = movie_preds(X_test, predicted_labels) # movie -> genre (onehot)
        label_dict = movie_preds(X_test, np.argmax(y_test, axis=1)) # movie -> genre (onehot)

        print('Test Accuracy predicting Movie Genres: ', test_accuracy(pred_dict, label_dict)) # accuracy
        #
        pred_dict, label_dict = convert_onehot_to_genre(pred_dict, label_dict, num_to_genre)

        print('Movie\tPredicted\tActual')

        for mov in pred_dict.keys():
            print("%s\t%s\t%s" % (mov, pred_dict[mov], label_dict[mov]))


if __name__ == "__main__":
    # Uncomment this to use your GPU

    # with tf.device('/gpu:0'):
    #     gpu_available = tf.test.is_gpu_available()
    #     print("GPU Available: ", gpu_available)
    #     run_model(multiclass=False)

    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)
    run_model(multiclass=True)
