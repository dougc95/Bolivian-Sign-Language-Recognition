import cv2
import time
import math
import os

import numpy as np
import pandas as pd
import holisticModule as hm


class Preprocessor:
    def __init__(self):
        self.matrix = np.zeros(42)

    def add_columns(self, feat_vector):
        feat_vector = np.array(feat_vector)
        self.matrix = np.column_stack((self.matrix, feat_vector))

    def cvt2csv(self, file_path, file):
        print("PRUEBA 2")
        file = os.path.basename(file)[0:-4]
        file_name = file_path+file+'.csv'
        print(file_name)
        df = pd.DataFrame(self.matrix)
        if self.matrix.shape[1] > 1:
            df = df.iloc[:, 1:]  # Drops first column
        print("PRUEBA")
        print(file_name)
        df.to_csv(file_name, index=True)

    def convert2csv(self, file_path, file):
        file = os.path.basename(file)[0:-4]
        file_name = file_path+"/"+file+'.csv'
        df = pd.DataFrame(self.matrix)
        if self.matrix.shape[1] > 1:
            df = df.iloc[:, 1:]  # Drops first column
        df.to_csv(file_name, index=True)

    # p2 is ref point, If you want the the angle between the line defined by these two points and the horizontal axis
    def get_angleX(self, p1, p2):
        return int(np.round(np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / np.pi))

    def get_distance(self, p1, p2):
        coordinate_distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return int(coordinate_distance)

    # Calculate distance matrix from an specific point(in this case NOSE) to the other points of interest
    def get_distMat(self, matrixHolistic, matrixLeft, matrixRight):
        distList = []
        NOSE = matrixHolistic[0]
        for i in matrixHolistic:
            distList.append({self.get_distance(NOSE, i), self.get_angleX(NOSE, i)})
        for i in matrixLeft:
            distList.append({self.get_distance(NOSE, i), self.get_angleX(NOSE, i)})
        for i in matrixRight:
            distList.append({self.get_distance(NOSE, i), self.get_angleX(NOSE, i)})
        # print('Matriz de Dist')
        # print(distList)
        # print('Matriz de Dist - NOSE')
        # print(distList[1:])
        # print(f"El tamaño de DISTANCIA: {len(distList)}")
        return distList[1:]


def main():
    # Check current directory
    cwd = os.getcwd()
    # Check and create data directory
    if not (os.path.exists(f'{cwd}/data')):
        os.mkdir(f'{cwd}/data')
    DATA = cwd+'/data/'
    print(DATA)
    # Iterate through videos
    DATASET_PATH = '/home/d3m1ur60/Desktop/LSBv2/'
    PATH1 = '/home/d3m1ur60/Desktop/LSBv2/Ayuda/ayuda_V1-0002.mp4'
    PATH2 = '/home/d3m1ur60/Desktop/LSBv2/Bolivia/bolivia_V1-0033.mp4'
    cap = cv2.VideoCapture(PATH2)
    pastTime = 0
    detector = hm.HolisticDetector()
    counter = 0
    # TEST
    prep = Preprocessor()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call Detection
        detector.find_pose(frame)  # Must be called for each detection
        detector.get_lm()
        detector.draw_pose(frame)  # Optional, just for drawing
        mat1, mat2, mat3 = detector.get_matrix()  # Returns holistic,left hand,right hand
        feat_vec = prep.get_distMat(mat1, mat2, mat3)
        prep.add_columns(feat_vec)
        # Get FPS
        counter = counter + 1
        currentTime = time.time()
        fps = int(1 / (currentTime - pastTime))
        pastTime = currentTime
        # Draw INFO
        cv2.putText(frame, f'FPS:{fps}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame, f'COUNT:{counter}', (350, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # Display
        cv2.imshow("original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f'TOTAL: {counter}')
    print('INTENTO ')
    print(prep.matrix)
    print(prep.matrix.shape)
    print(DATA)
    prep.cvt2csv(DATA, PATH1)


if __name__ == '__main__':
    main()
