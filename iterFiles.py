import cv2
import time
import math
import os

import numpy as np
import pandas as pd
import holisticModule as hm
import prepareData as prep


def create_dataFolders(self, CATEGORY, DATADIR, LABEL):
    detector = hm.HolisticDetector()
    for category in CATEGORY:
        preprocess = prep.Preprocessor()
        path = os.path.join(DATADIR, category)
        lblPath = os.path.join(LABEL, category)
        for vid in os.listdir(path):
            cap = cv2.VideoCapture(vid)
            ret, frame = cap.read()
            while ret:
                ret, frame = cap.read()
                detector.find_pose(frame)
                detector.get_lm()
                mat1, mat2, mat3 = detector.get_matrix()
                feat_vec = prep.get_distMat(mat1, mat2, mat3)
                preprocess.add_columns(feat_vec)
            #preprocess.cvt2csv(lblPath,vid)
            break
        break


def main():
    DATADIR = '/home/d3m1ur60/Desktop/LSBv2/'
    CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
                'Necesito', 'Quien Occidente', 'Quien Oriente', 'Saludos']
    cwd = os.getcwd()
    if not (os.path.exists(f'{cwd}/labelsLSB')):
        os.mkdir(f'{cwd}/labelsLSB')
    LABELS = f'{cwd}/labels'
    create_dataFolders(CATEGORY, DATADIR, LABELS)



if __name__ == "__main__":
    main()
