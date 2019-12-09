import tensorflow as tf
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from preprocess import load_framedata
from dataGenerator import dataGenerator


def setup_model(shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
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
    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam() # tune adam parameters possibly
    model.compile(loss='categorical_crossentropy', optimizer=adam)
    model.summary()

    return model

def run_model():
    X_train, X_test, y_train, y_test = load_framedata() # loads in data from preprocess

    train_generator = dataGenerator(X_train, y_train, batch_size=128) # see datagenerator class

    model = setup_model((128, 176), num_classes=24)
    model.fit_generator(train_generator, epochs=3, validation_split=0.1, verbose=1) # 10% of train is validation data

    model.save('model_1')

    print('------- Testing model -------')
    
    score = model.evaluate(X_test, y_test, batch_size=128)
    print("Test Metrics: ", score)

if __name__ == "__main__":
    # setup_model()
    run_model()