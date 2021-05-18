import cv2
import time
import math
import numpy as np
import pandas as pd
import holisticModule as hm


def get_distance(p1, p2):
    coordinate_distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return int(coordinate_distance)


# Calculate distance matrix from an specific point(in this case NOSE) to the other points of interest
def get_distMat(matrixHolistic, matrixLeft, matrixRight):
    distList = []
    NOSE = matrixHolistic[0]
    for i in matrixHolistic:
        distList.append(get_distance(NOSE, i))
    for i in matrixLeft:
        distList.append(get_distance(NOSE, i))
    for i in matrixRight:
        distList.append(get_distance(NOSE, i))
    print('Matriz de Dist')
    print(distList)
    print('Matriz de Dist - NOSE')
    print(distList[1:])
    print(f"El tamaño de DISTANCIA: {len(distList)}")
    return distList[1:]


class Organize:
    def __init__(self):
        self.matrix = np.zeros(42)

    def add_columns(self, feat_vector):
        feat_vector = np.array(feat_vector)
        self.matrix = np.column_stack((self.matrix, feat_vector))

    def cvt2csv(self):
        


def main():
    PATH1 = '/home/d3m1ur60/Desktop/LSBv2/Ayuda/ayuda_V1-0002.mp4'
    PATH2 = '/home/d3m1ur60/Desktop/LSBv2/Bolivia/bolivia_V1-0032.mp4'
    cap = cv2.VideoCapture(PATH2)
    pastTime = 0
    detector = hm.HolisticDetector()
    counter = 0
    # TEST
    prep = Organize()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call Detection
        detector.find_pose(frame)  # Must be called for each detection
        detector.get_lm()
        detector.draw_pose(frame)  # Optional, just for drawing
        mat1, mat2, mat3 = detector.get_matrix()  # Returns holistic,left hand,right hand
        feat_vec = get_distMat(mat1, mat2, mat3)
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


if __name__ == '__main__':
    main()
