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
DATADIR2 = 'C:\\Users\\DougC\\PycharmProjects\\Bolivian-Sign-Language-Recognition\\logs'
CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']

dictionary = {
    'Ayuda': 0,
    'Bolivia': 1,
    'Como': 2,
    'Dinero': 3,
    'Doctor': 4,
    'Donde': 5,
    'Explicacion': 6,
    'Guardar': 7,
    'Necesito': 8,
    'Quien Occidente': 9,
    'Saludos': 10,
}
N_TIME_STEPS = 272


def get_dataFolders(CATEGORY, DATADIR, dict):
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
            # aux = np.stack((mag, ang), axis=2)
            aux = np.vstack((mag, ang))
            segments.append(aux)
            #labels.append(dictionary[category])
            labels.append(category)
    print(f"shape interno: {np.array(segments).shape}")
    return labels, segments


labels, segments = get_dataFolders(CATEGORY, DATADIR,dictionary)
labels = np.asarray(pd.get_dummies(labels), dtype=int)
print(np.array(labels).shape)
# Prepare input
train_x, valid_x, train_y, valid_y = train_test_split(segments, labels, test_size=0.1, random_state=42)
train_x = np.array(train_x)
valid_x = np.array(valid_x)
train_y = np.array(train_y)
valid_y = np.array(valid_y)

# Model params
EPOCHS = 20
BATCH_SIZE = 16
NAME = f"{EPOCHS}-SEQ-PRED-{int(time.time())}"

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

model.add(Dense(11, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir=f"{DATADIR2}\\{NAME}")
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.2f}"  # file that will have the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models\\{}.model".format(filepath, monitor='val_accuracy',
                                                       verbose=1, save_best_only=True,
                                                       mode='max'))  # saves only the best ones

history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(valid_x, valid_y), callbacks=[tensorboard, checkpoint])
