import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

PATH_MODEL = 'C:\\Users\\DougC\\PycharmProjects\\Bolivian-Sign-Language-Recognition\\models\\RNN_Final-05-0.573.model'
PATH_VIDEO = 'C:\\Users\\DougC\\PycharmProjects\\Bolivian-Sign-Language-Recognition\\labelsLSB\\Ayuda\\ayuda_V1-0001.csv'


def getData(file):
    segments = []
    data = pd.read_csv(file)
    while data.shape[1] < 272:
        data = np.column_stack((data, np.zeros(42)))
        pass
    data = pd.DataFrame(data)
    mag = data.iloc[:, ::2]
    ang = data.iloc[:, 1::2]
    aux = np.vstack((mag, ang))
    segments.append(aux)
    return np.array(segments)

data = getData(PATH_VIDEO)
print(data.shape)
new_model = tf.keras.models.load_model(PATH_MODEL)
print(new_model.summary())
print(f"Video corresponde a {new_model.predict_classes(data)}")

