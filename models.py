import os
import numpy as np
import pandas as pd
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, ConvLSTM2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow.keras.preprocessing

from sklearn import preprocessing
from collections import deque


def get_dataFolders(CATEGORY, DATADIR):
    N_TIME_STEPS = 50
    N_FEATURES = 1

    # Get X and Y indepents DONE
    # Transpose
    # segments.append([xs, ys]) DONE
    # labels.append(category) DONE
    # labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

    step = 2
    segments = []
    labels = []
    count = []
    for category in CATEGORY:
        # print(f'La categoria actual: {category}')
        path = os.path.join(DATADIR, category)
        for name in os.listdir(path):
            matDist = []
            matAng = []
            # print(f"El file a rep es: {name}")
            file = os.path.join(path, name)
            df = pd.read_csv(file).to_numpy()
            count.append(df.shape[1])
            # Pad sequence
            if df.shape[1] < N_TIME_STEPS:
                dif = N_TIME_STEPS - df.shape[1]
                np.pad(df, (dif // 2, dif // 2), 'constant', constant_values=(0, 0))
            # Format data
            for i in range(0, df.shape[1], step):
                mag = df[:, i].astype(int)
                ang = df[:, i + 1].astype(int)
                matDist.append([mag])
                matAng.append([ang])
                label = category

                segments.append([mag[0,:],mag[1,:],mag[2,:],mag[3,:],mag[4,:],mag[5,:],mag[6,:],mag[7,:],mag[8,:],mag[9,:],mag[10,:],mag[11,:],mag[12,:],mag[13,:],
                                 mag[14,:],mag[15,:],mag[16,:],mag[17],mag[18,:],mag[19,:],mag[20,:],mag[21,:],mag[22,:],mag[23,:],mag[24,:],mag[25,:],mag[26,:],mag[27,:],
                                 mag[28,:],mag[29,:],mag[30,:],mag[31,:],mag[32,:],mag[33,:],mag[34,:],mag[35,:],mag[36,:],mag[37,:],mag[38,:],mag[39,:],mag[40,:],mag[41,:]])
            matDist = np.transpose(matDist)
            matDist = np.transpose(matAng)
            labels.append(category)
            print(f"COUNT: {count}")
            print(len(segments))
            # print(aux)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    print(f'El maximo de frame es:{np.argmax(count)}')
    print(f'El maximo de frame es:{max(count)}')

    return labels, segments


DATADIR = 'C:/Users/DougC/PycharmProjects/Bolivian-Sign-Language-Recognition/labelsLSB'
CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Saludos']


labels, segments = get_dataFolders(CATEGORY, DATADIR)
N_TIME_STEPS = 272
N_FEATURES = 42*2
reshaped_segments = np.asarray(segments, dtype=np.int).reshape(-1, N_TIME_STEPS, N_FEATURES)

print(np.array(segments).shape)

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
