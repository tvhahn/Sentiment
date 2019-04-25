import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

# prevent tensorflow from using GPU. Otherwise, run out of memory
# https://stackoverflow.com/questions/44552585/prevent-tensorflow-from-accessing-the-gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# import tensorflow_hub as hub
import tensorflow as tf

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional,Input, Embedding, Flatten
from keras import Model
from keras.optimizers import RMSprop
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# Load Pickles
pickle_path = '/home/tim/Documents/Sentiment/Data/pickles'

#x_train
pickle_in = open(os.path.join(pickle_path,'x_train.pickle'),"rb")
x_train = pickle.load(pickle_in)

#y_train
pickle_in = open(os.path.join(pickle_path,'y_train.pickle'),"rb")
y_train = pickle.load(pickle_in)

#x_val
pickle_in = open(os.path.join(pickle_path,'x_dev.pickle'),"rb")
x_val = pickle.load(pickle_in)

#y_val
pickle_in = open(os.path.join(pickle_path,'y_dev.pickle'),"rb")
y_val = pickle.load(pickle_in)

#x_test
pickle_in = open(os.path.join(pickle_path,'x_test.pickle'),"rb")
x_test = pickle.load(pickle_in)

#y_test
pickle_in = open(os.path.join(pickle_path,'y_test.pickle'),"rb")
y_test = pickle.load(pickle_in)

# convert y values to one-hot encoding
y_train = to_categorical(y_train - y_train.min())
y_val = to_categorical(y_val - y_val.min())
y_test = to_categorical(y_test - y_test.min())

# build model
lstm_1_units = [150]
lstm_2_units = lstm_1_units
lstm_layers = [0,1]
dense_1_units = [150]
dense_2_unit = 7
dense_layers = [1]
drop_rates = [0.5]
batch_sizes = [512]
lstm_activations = ['relu']
learn_rates = [1e-4,1e-3,1e-2]
decay_rates = [0.0,1e-4]

for lstm_1_unit in lstm_1_units:
        for lstm_layer in lstm_layers:
                for dense_1_unit in dense_1_units:
                        for dense_layer in dense_layers:
                                for drop_rate in drop_rates:
                                        for lstm_activation in lstm_activations:
                                                for batch_size in batch_sizes:
                                                        for learn_rate in learn_rates:
                                                                for decay_rate in decay_rates:
                                                                        model = Sequential()
                                                                        model.add(Bidirectional(LSTM(lstm_1_unit, activation=lstm_activation, return_sequences=True,dropout=drop_rate), input_shape=(80,512)))

                                                                        for i in range(lstm_layer):
                                                                                model.add(Bidirectional(LSTM(lstm_1_unit, activation=lstm_activation, return_sequences=True,dropout=drop_rate)))

                                                                        model.add(Bidirectional(LSTM(lstm_1_unit,activation=lstm_activation, return_sequences=False,dropout=drop_rate)))
                                                                        

                                                                        for j in range(dense_layer):
                                                                                model.add(Dense(dense_1_unit, activation = 'relu'))
                                                                                model.add(Dropout(drop_rate))

                                                                        model.add(Dense(dense_2_unit, activation = 'softmax'))
                                                                        model.compile(optimizer=(RMSprop(lr=learn_rate, rho=0.9, decay=decay_rate)),loss='categorical_crossentropy', metrics=['accuracy'])

                                                                        param_size = model.count_params()

                                                                        # TRAIN
                                                                        # model name
                                                                        NAME = "biLSTM_P{}_L1={}_L2={}_LSTM-count{}_d1={}_drop={}_bsize={}_act={}_dense={}_lr={}_decay={}_{}".format(param_size,
                                                                                                                        lstm_1_unit,
                                                                                                                        lstm_1_unit,
                                                                                                                        lstm_layer+2,
                                                                                                                        dense_1_unit,
                                                                                                                        drop_rate,
                                                                                                                        batch_size,
                                                                                                                        lstm_activation,
                                                                                                                        dense_layer,
                                                                                                                        learn_rate,
                                                                                                                        decay_rate,
                                                                                                                        int(time.time()))

                                                                        print("######################################################")
                                                                        print(NAME)
                                                                        print(model.summary())

                                                                        # callbacks
                                                                        tensorboard = TensorBoard(log_dir="logs/biLSTM1/{}".format(NAME))
                                                                        early_stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min',restore_best_weights=True)
                                                                        

                                                                        path = "/home/tim/Documents/Sentiment/models"
                                                                        # os.mkdir(os.path.join(path, NAME))
                                                                        path = os.path.join(path, NAME)

                                                                        save_path2 = (path + "/epoch.{epoch:02d}-val_loss.{val_loss:.2f}.h5")

                                                                        checkpoint2 = ModelCheckpoint(filepath=save_path2,monitor="val_loss",verbose=0,save_best_only=True,mode="min",)

                                                                        model.fit(x_train, y_train,
                                                                                epochs = 500,
                                                                                validation_data=(x_val,y_val),
                                                                                shuffle=True,
                                                                                batch_size=batch_size,
                                                                                verbose = 0,
                                                                                callbacks=[tensorboard,early_stop])

                                                                        model.save("models/{}.h5".format(NAME))
