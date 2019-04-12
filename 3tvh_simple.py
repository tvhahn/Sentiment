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
lstm_1_units = [100,50,20]
lstm_2_units = [100,50,20]
dense_1_units = [100,50,20]
dense_2_unit = 7
drop_rates = [0.5,0.2,0]
batch_sizes = [64,128,32,512]

for lstm_1_unit in lstm_1_units:
    for lstm_2_unit in lstm_2_units:
        for dense_1_unit in dense_1_units:
                for drop_rate in drop_rates:
                        for batch_size in batch_sizes:
                                model = Sequential()
                                model.add(Bidirectional(LSTM(lstm_1_unit, activation='relu', return_sequences=True), input_shape=(80,512)))
                                model.add(Bidirectional(LSTM(lstm_2_unit,activation='relu', return_sequences=False)))
                                model.add(Dense(dense_1_unit, activation = 'relu'))
                                model.add(Dropout(drop_rate))
                                model.add(Dense(dense_2_unit, activation = 'softmax'))
                                model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


                                # TRAIN
                                # model name
                                NAME = "biLSTM_L1={}_L2={}_d1={}_d2={}_drop={}_bsize={}_{}".format(lstm_1_unit,
                                                                                lstm_2_unit,
                                                                                dense_1_unit,
                                                                                dense_2_unit,
                                                                                drop_rate,
                                                                                batch_size,
                                                                                int(time.time()))

                                print("######################################################")
                                print(NAME)
                                print(model.summary())

                                # callbacks
                                tensorboard = TensorBoard(log_dir="logs/biLSTM1/{}".format(NAME))
                                early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min') 

                                b = batch_size # batch size

                                model.fit(x_train, y_train,
                                        epochs = 500,
                                        validation_data=(x_val,y_val),
                                        shuffle=True,
                                        batch_size=b,
                                        verbose = 0,
                                        callbacks=[tensorboard])
