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
    N_TIME_STEPS = 80
    N_FEATURES = 1
    step = 2
    segments = []
    labels = []
    for category in CATEGORY:
        print(f'La categoria actual: {category}')
        path = os.path.join(DATADIR, category)
        for name in os.listdir(path):
            print(f"El file a rep es: {name}")
            file = os.path.join(path, name)
            # print(f"El file_path a rep es: {file}")
            df = pd.read_csv(file)
            df = df.iloc[:, 1:]
            for i in range(0, len(df) - N_TIME_STEPS, step):
                x = df.iloc[i: i + N_TIME_STEPS]
                label = category
                segments.append([x])
                labels.append(label)
                print(x)
            # print(df.iloc[:,0])
            continue
        continue


DATADIR = 'C:/Users/DougC/PycharmProjects/Bolivian-Sign-Language-Recognition/labelsLSB'
CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
            'Necesito', 'Quien Occidente', 'Quien Oriente', 'Saludos']

get_dataFolders(CATEGORY,DATADIR)
