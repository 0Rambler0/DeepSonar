import multiprocessing
import os
import sys
import time
sys.path.append('/home/dslab/hx/audio_detection/DeepSonar/SR_module')
import utils as ut
import numpy as np
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from data_utils import *

def create_model():
    # inputs = Input(shape=(512, 1009))
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(512, 113)))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model

def train(model, train_set, train_labels, test_set, test_labels, batch_size, epochs, save_path):
    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    history = model.fit(train_set, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_set, test_labels))

    model.save_weights(os.path.join(save_path, 'weight.h5'))

    return history



    