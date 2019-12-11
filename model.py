import tensorflow as tf
import numpy as np
import pickle
import os
import sys

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from preprocess import load_framedata, split_on_movie, split_on_movie_normalized
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

    adam = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999) # tune adam parameters possibly
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
        pred = np.argmax(frame_preds[i])

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

def top_genre_accuracy(y_pred, y_true):
    sum = 0
    for i in range(y_true.shape[0]):
        curr_sum = np.sum(np.equal(y_pred[i],y_true[i]))
        curr_sum /=3
        sum += curr_sum
    return sum / y_true.shape[0]

"""
Return top 3 predicted genres from probabilities as one-hot labels
"""

def probs_to_preds(probs):
    preds = np.zeros_like(probs)
    for i in range(probs.shape[0]):
        top3 = probs[i].argsort()[-3:][::-1]
        temp = np.zeros(probs.shape[1])
        pred = np.put(temp, top3, 1)
        preds[i] = pred
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

def run_model(multiclass=True, normalized=True):
    # X_train, X_test, y_train, y_test, encoder = load_framedata(multiclass) # loads in data from preprocess
    num_classes=11

    if normalized:
        X_train, X_test, y_train, y_test, encoder = split_on_movie_normalized()
    else:
        X_train, X_test, y_train, y_test, encoder = split_on_movie(multiclass=multiclass)
        if multiclass:
            num_classes=28
        else:
            num_classes=24
    # Used to convert from onehot labels back to genre strings

    # print(y_test)
    # print(sum(y_test))

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
    # test_generator = dataGenerator(X_test[0:20], y_test[0:20], batch_size=32)
    # img, label = test_generator.__getitem__(0)
    # print(img[0])
    # print(np.min(img[0]))
    # print(X_test[0:20])
    # print(y_test[0:20])

    try:
        os.stat('./model_14.h5')
        print("existing model found; loading model...")
        model = load_model('./model_1.h5')
    except:
        print("no preloaded model. training model...")
        if multiclass:
            model = setup_model((128, 176), num_classes=num_classes)
        else:
            model = setup_model((128, 176), num_classes=num_classes)
        model.fit_generator(train_generator, epochs=1, verbose=1)
        print("saving model...")
        model.save('model_1.h5')

    print('------- Testing model -------')

    # score = model.evaluate_generator(test_generator, verbose=1)
    # exit()
    # print("Test Metrics: ", score)
    # np.set_printoptions(threshold=sys.maxsize)

    frame_preds = model.predict_generator(test_generator, verbose=1) # generate prediction labels on the frames

    # print(frame_preds.shape)
    # print(frame_preds)
    # print(np.argmax(frame_preds, axis=1))

    if multiclass:
        # categorical accuracy
        acc = tf.keras.metrics.CategoricalAccuracy()
        # idxs = np.argsort(frame_preds)[::-1][:3]
        acc.update_state(frame_preds, y_test)
        result = acc.result()
        print("categorical accuracy: ")
        print(result)

        # hamming score
        preds = probs_to_preds(frame_preds)
        score = hamming_score(preds, y_test)
        print("hamming score: ")
        print(score)

        # top genre only
        top = top_genre_accuracy(preds, y_test)
        print("top 3 genre accuracy: ")
        print(top)
    else:
        pred_dict = movie_preds(X_test, frame_preds.flatten()) # movie -> genre (onehot)
        label_dict = movie_preds(X_test, y_test) # movie -> genre (onehot)

        print('Test Accuracy predicting Movie Genres: ', test_accuracy(pred_dict, label_dict)) # accuracy
        #
        pred_dict, label_dict = convert_onehot_to_genre(pred_dict, label_dict, num_to_genre)
        
        print('Movie\tPredicted\tActual')
        
        for mov in pred_dict.keys():
            print("%s\t%s\t%s" % (mov, pred_dict[mov], label_dict[mov]))


if __name__ == "__main__":
    # setup_model()

    # Kyle uncomment this to use your GPU

    # with tf.device('/gpu:0'):
    #     gpu_available = tf.test.is_gpu_available()
    #     print("GPU Available: ", gpu_available)
    #     run_model(multiclass=False)

    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)
    run_model(multiclass=False)
