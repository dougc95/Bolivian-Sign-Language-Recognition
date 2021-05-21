import os
import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn import preprocessing
from collections import deque
from sklearn.model_selection import train_test_split

DATADIR = 'C:/Users/DougC/PycharmProjects/Bolivian-Sign-Language-Recognition/labelsLSB'
CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']
N_TIME_STEPS = 272


def get_dataFolders(CATEGORY, DATADIR):
    labels = []
    segments = []
    for category in CATEGORY:
        path = os.path.join(DATADIR, category)
        for name in os.listdir(path):
            file = os.path.join(path, name)
            data = pd.read_csv(file)
            while data.shape[1] < 272:
                data = np.column_stack((data, np.zeros(42)))
            data = pd.DataFrame(data)
            mag = data.iloc[:, ::2]
            ang = data.iloc[:, 1::2]
            aux = np.stack((mag, ang), axis=2)
            segments.append(aux)
            labels.append(category)
    return labels, segments


labels, segments = get_dataFolders(CATEGORY, DATADIR)
print(np.array(segments).shape)
# reshaped_segments = np.asarray(segments, dtype=int).reshape(-1, , )
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
print(np.array(labels).shape)
# WEA
train_x, test_y, train_y, test_y = train_test_split(segments, labels, test_size=0.2)
train_x = np.array(train_x)

# Model params
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{N_TIME_STEPS}-SEQ-PRED-{int(time.time())}"

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # file that will have the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                                      mode='max'))  # saves only the best ones

history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint])
