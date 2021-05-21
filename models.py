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


def get_dataFolders(CATEGORY, DATADIR):
    N_TIME_STEPS = 50
    N_FEATURES = 1

    # Get X and Y indepents
    # Transpose
    # segments.append([xs, ys])
    # labels.append(category)
    # labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)

    step = 1
    segments = []
    labels = []
    for category in CATEGORY:
        # print(f'La categoria actual: {category}')
        path = os.path.join(DATADIR, category)
        for name in os.listdir(path):
            # print(f"El file a rep es: {name}")
            file = os.path.join(path, name)
            df = pd.read_csv(file).to_numpy()
            for i in range(0, df.shape[1]-1, step):
                mag = df[:, i].astype(int)
                ang = df[:, i + 1].astype(int)
                label = category
                segments.append([mag, ang])
                labels.append(label)
            print(len(segments))
    return labels, segments


DATADIR = 'C:/Users/DougC/PycharmProjects/Bolivian-Sign-Language-Recognition/labelsLSB'
CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Quien Oriente', 'Saludos']

N_TIME_STEPS = 48*
N_FEATURES = 42*2
labels, segments = get_dataFolders(CATEGORY, DATADIR)
reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)

print(np.array(segments).shape)
