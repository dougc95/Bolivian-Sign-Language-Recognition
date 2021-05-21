import cv2
import time
import math
import os

import numpy as np
import pandas as pd
import holisticModule as hm
import prepareData as prep


def create_dataFolders(CATEGORY, DATADIR, LABEL):
    detector = hm.HolisticDetector()
    for category in CATEGORY:
        print(f'La categoria actual: {category}')
        path = os.path.join(DATADIR, category)
        lblPath = os.path.join(LABEL, category)
        print(f'La etiqueta actual: {lblPath}')
        if not (os.path.exists(lblPath)):
            os.mkdir(lblPath)
        for vid in os.listdir(path):
            preprocess = prep.Preprocessor()
            print(f"El video a rep es: {vid}")
            cap = cv2.VideoCapture(f"{DATADIR}/{category}/{vid}")
            ret, frame = cap.read()
            while ret:
                ret, frame = cap.read()
                if not ret:
                    continue
                detector.find_pose(frame)
                detector.get_lm()
                mat1, mat2, mat3 = detector.get_matrix()
                feat_vec = preprocess.get_distMat(mat1, mat2, mat3)
                preprocess.add_columns(feat_vec)

            preprocess.convert2csv(lblPath, vid)
            continue
        continue


def main():
    # DATADIR = '/home/d3m1ur60/Desktop/LSBv2/'
    DATADIR = 'C:/Users/DougC/Desktop/LSBv2'
    CATEGORY = ['Ayuda', 'Bolivia', 'Como', 'Dinero', 'Doctor', 'Donde', 'Explicacion', 'Guardar',
                'Necesito', 'Quien Occidente', 'Quien Oriente', 'Saludos']
    cwd = os.getcwd()
    if not (os.path.exists(f'{cwd}/labelsLSB')):
        os.mkdir(f'{cwd}/labelsLSB')
    LABELS = f'{cwd}/labelsLSB'
    create_dataFolders(CATEGORY, DATADIR, LABELS)


if __name__ == "__main__":
    main()
