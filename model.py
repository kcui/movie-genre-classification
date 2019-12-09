import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from preprocess import load_framedata
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
    model.add(Dense(num_classes, activation='softmax'))

    adam = Adam() # tune adam parameters possibly
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

def run_model():
    X_train, X_test, y_train, y_test = load_framedata() # loads in data from preprocess

    train_generator = dataGenerator(X_train, y_train, batch_size=128) # see datagenerator class
    test_generator = dataGenerator(X_test, y_test, batch_size=128)

    model = setup_model((128, 176), num_classes=24)
    model.fit_generator(train_generator, epochs=1, verbose=1)

    model.save('model_1')

    print('------- Testing model -------')

    score = model.evaluate_generator(test_generator)
    print("Test Metrics: ", score)

if __name__ == "__main__":
    # setup_model()
    gpu_available = tf.test.is_gpu_available()
    print("GPU Available: ", gpu_available)
    run_model()
