import os

import numpy
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import holisticModule as hm
import prepareData as prep
import numpy as np
import cv2

PATH_MODEL = 'E:\\Backup Test1\\models\\RNN_Final-98-0.43.model'
PATH_VIDEO = 'C:\\Users\\DougC\\Desktop\\LSBv2\\Ayuda\\ayuda_V1-0001.mp4'


def main():
    new_model = tf.keras.models.load_model(PATH_MODEL)
    print(new_model.summary())
    detector = hm.HolisticDetector()
    prepare = prep.Preprocessor()
    cap = cv2.VideoCapture(PATH_VIDEO)  # PATH_VIDEO or CAMERA_FEED
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        detector.find_pose(frame)
        detector.get_lm()
        mat1, mat2, mat3 = detector.get_matrix()
        feat_vec = prepare.get_distMat(mat1, mat2, mat3)
        feat_vec = np.array(feat_vec)
        aux = numpy.concatenate((feat_vec[:, 0], feat_vec[:, 1]))

        print(f"Video corresponde a {new_model.predict_classes(())}")

if __name__ == '__main__':
    main()
