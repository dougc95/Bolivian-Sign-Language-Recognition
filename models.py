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


def get_dataFolders(CATEGORY, DATADIR):
    N_TIME_STEPS = 272
    N_FEATURES = 42
    DIMENSION = 2
    step = 2
    segments = np.empty(shape=(2,N_FEATURES,N_TIME_STEPS))
    labels = []
    for category in CATEGORY:
        # print(f'La categoria actual: {category}')
        path = os.path.join(DATADIR, category)
        for name in os.listdir(path):
            matDist = []
            matAng = []
            # print(f"El file a rep es: {name}")
            file = os.path.join(path, name)
            df = pd.read_csv(file).to_numpy()
            # Pad sequence
            if df.shape[1] < N_TIME_STEPS:
                dif = N_TIME_STEPS - df.shape[1]
                np.pad(df, (dif // 2, dif // 2), 'constant', constant_values=(0, 0))
            # Format data
            for i in range(0, df.shape[1], step):
                mag = df[:, i].astype(int)
                ang = df[:, i + 1].astype(int)  # trans matDist = np.transpose(matDist) matAng = np.transpose(matAng)
                matDist.append([mag])
                matAng.append([ang])
            matDist = np.transpose(matDist)
            #print(f'shape matDst {matDist}')
            matAng = np.transpose(matAng)
            np.append(segments, ([matDist[0], matDist[1], matDist[2], matDist[3], matDist[4],
                                  matDist[5], matDist[6], matDist[7], matDist[8], matDist[9],
                                  matDist[10], matDist[11], matDist[12], matDist[13],
                                  matDist[14], matDist[15], matDist[16], matDist[17], matDist[18],
                                  matDist[19], matDist[20], matDist[21], matDist[22], matDist[23],
                                  matDist[24], matDist[25], matDist[26], matDist[27],
                                  matDist[28], matDist[29], matDist[30], matDist[31], matDist[32],
                                  matDist[33], matDist[34], matDist[35], matDist[36], matDist[37], matDist[38],
                                  matDist[39], matDist[40], matDist[41],
                                  matAng[0], matAng[1], matAng[2], matAng[3], matAng[4], matAng[5],
                                  matAng[6], matAng[7], matAng[8], matAng[9], matAng[10], matAng[11],
                                  matAng[12], matAng[13],
                                  matAng[14], matAng[15], matAng[16], matAng[17], matAng[18],
                                  matAng[19], matAng[20], matAng[21], matAng[22], matAng[23],
                                  matAng[24], matAng[25], matAng[26], matAng[27],
                                  matAng[28], matAng[29], matAng[30], matAng[31], matAng[32],
                                  matAng[33], matAng[34], matAng[35], matAng[36], matAng[37],
                                  matAng[38], matAng[39], matAng[40], matAng[41]]))
            labels.append(category)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
    aux = np.array(segments)
    print(aux.shape)
    return labels, segments


labels, segments = get_dataFolders(CATEGORY, DATADIR)
N_TIME_STEPS = 272
N_FEATURES = 42 + 42

# WEA
train_x, test_y, train_y, test_y = train_test_split(segments, labels, test_size=0.2)

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
