import cv2
import time
import math
import mediapipe as mp


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
    return distList


class HolisticDetector:
    def __init__(self, static_image_mode=False, model_complexity=True, smooth_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Same params required for Holistic
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        # Create mp object, then the holistic module and the drawing toolqqs
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(self.static_image_mode, self.model_complexity,
                                                 self.smooth_landmarks, self.min_detection_confidence,
                                                 self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    # Model treats img as RGB, next function should be run on every frame
    def find_pose(self, frame):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.height, self.width, _ = frame.shape
        self.results = self.holistic.process(imgRGB)

    # Transform relative position from model to coordinates
    def get_lm(self):
        self.lmListBody = []
        self.lmListLeft = []
        self.lmListRight = []

        if self.results.pose_landmarks:
            body = self.results.pose_landmarks
            for number, lm in enumerate(body.landmark):
                cx, cy = int(lm.x * self.width), int(lm.y * self.height)
                self.lmListBody.append([number, cx, cy])

        if self.results.left_hand_landmarks:
            left = self.results.left_hand_landmarks
            for number, lm in enumerate(left.landmark):
                cx, cy = int(lm.x * self.width), int(lm.y * self.height)
                self.lmListLeft.append([number, cx, cy])

        if self.results.right_hand_landmarks:
            right = self.results.right_hand_landmarks
            for number, lm in enumerate(right.landmark):
                cx, cy = int(lm.x * self.width), int(lm.y * self.height)
                self.lmListRight.append([number, cx, cy])

    def draw_pose(self, frame):
        self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpHolistic.UPPER_BODY_POSE_CONNECTIONS)
        self.mpDraw.draw_landmarks(frame, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
        self.mpDraw.draw_landmarks(frame, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
        return frame

    # Params 'body', 'left', 'right'
    def get_angle(self, p1, p2, p3, section='body'):
        if section == 'body':
            lmList = self.lmListBody
        if section == 'left':
            lmList = self.lmListLeft
        if section == 'right':
            lmList = self.lmListRight
        # Get points
        _, x1, y1 = lmList[p1]
        _, x2, y2 = lmList[p2]
        _, x3, y3 = lmList[p3]
        # Get angle
        angle = abs(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))
        return angle

    def get_matrix(self):
        # Init Left and Right Matrices
        matrixLeft = [[0, 0]] * 15
        matrixRight = [[0, 0]] * 15
        matrixHolistic = [[0, 0]] * 12
        if self.lmListBody:
            pt0 = self.lmListBody[0][1:]  # Normalization pt NOSE
            matrixLeft = [pt0] * 15  # FOR DIST
            matrixRight = [pt0] * 15  # FOR DIST
            pt1 = self.lmListBody[12][1:]  # Right Shoulder
            pt2 = self.lmListBody[11][1:]  # Left Shoulder
            pt3 = self.lmListBody[14][1:]  # Right Elbow
            pt4 = self.lmListBody[13][1:]  # Left Elbow
            pt5 = self.lmListBody[16][1:]  # Right Wrist
            pt6 = self.lmListBody[15][1:]  # Left Wrist
            # matrixHolistic = np.array([pt1, pt2, pt3, pt4, pt5, pt6])
            aux_pt1 = self.lmListBody[22][1:]  # Right Thumb
            aux_pt2 = self.lmListBody[21][1:]  # Left Thumb
            aux_pt3 = self.lmListBody[20][1:]  # Right Index
            aux_pt4 = self.lmListBody[19][1:]  # Left Index
            aux_pt5 = self.lmListBody[18][1:]  # Right Pinky
            aux_pt6 = self.lmListBody[17][1:]  # Left Pinky
            # matrixAuxHolistic = np.array([aux_pt1, aux_pt2, aux_pt3, aux_pt4, aux_pt5, aux_pt6])
            matrixHolistic = [pt0, pt1, pt2, pt3, pt4, pt5, pt6, aux_pt1, aux_pt2, aux_pt3, aux_pt4, aux_pt5, aux_pt6]
            if self.lmListLeft:
                # Section 1x
                left_pt1 = self.lmListLeft[2][1:]  # THUMB MCP
                left_pt2 = self.lmListLeft[5][1:]  # INDEX MCP
                left_pt3 = self.lmListLeft[9][1:]  # MIDDLE MCP
                left_pt4 = self.lmListLeft[13][1:]  # RING MCP
                left_pt5 = self.lmListLeft[17][1:]  # PINKY MCP
                # Section 2x
                left_pt6 = self.lmListLeft[3][1:]  # THUMB IP
                left_pt7 = self.lmListLeft[6][1:]  # INDEX IP
                left_pt8 = self.lmListLeft[10][1:]  # MIDDLE IP
                left_pt9 = self.lmListLeft[14][1:]  # RING IP
                left_pt10 = self.lmListLeft[18][1:]  # PINKY IP
                # Section 3x
                left_pt11 = self.lmListLeft[4][1:]  # THUMB TIP
                left_pt12 = self.lmListLeft[8][1:]  # INDEX TIP
                left_pt13 = self.lmListLeft[12][1:]  # MIDDLE TIP
                left_pt14 = self.lmListLeft[16][1:]  # RING TIP
                left_pt15 = self.lmListLeft[20][1:]  # PINKY TIP
                matrixLeft = [left_pt1, left_pt2, left_pt3, left_pt4, left_pt5,
                              left_pt6, left_pt7, left_pt8, left_pt9, left_pt10,
                              left_pt11, left_pt12, left_pt13, left_pt14, left_pt15]
            if self.lmListRight:
                # Section 1x
                right_pt1 = self.lmListRight[2]  # THUMB MCP
                right_pt2 = self.lmListRight[5]  # INDEX MCP
                right_pt3 = self.lmListRight[9]  # MIDDLE MCP
                right_pt4 = self.lmListRight[13]  # RING MCP
                right_pt5 = self.lmListRight[17]  # PINKY MCP
                # Section 2x
                right_pt6 = self.lmListRight[3]  # THUMB IP
                right_pt7 = self.lmListRight[6]  # INDEX IP
                right_pt8 = self.lmListRight[10]  # MIDDLE IP
                right_pt9 = self.lmListRight[14]  # RING IP
                right_pt10 = self.lmListRight[18]  # PINKY IP
                # Section 3x
                right_pt11 = self.lmListRight[4]  # THUMB TIP
                right_pt12 = self.lmListRight[8]  # INDEX TIP
                right_pt13 = self.lmListRight[12]  # MIDDLE TIP
                right_pt14 = self.lmListRight[16]  # RING TIP
                right_pt15 = self.lmListRight[20]  # PINKY TIP
                matrixRight = [right_pt1, right_pt2, right_pt3, right_pt4, right_pt5,
                               right_pt6, right_pt7, right_pt8, right_pt9, right_pt10,
                               right_pt11, right_pt12, right_pt13, right_pt14, right_pt15]
        # print(f"Matriz Holistic \n {matrixHolistic}")
        # print(f"Matriz Izq \n {matrixLeft}")
        # print(f"Matriz Der \n {matrixRight}")
        return matrixHolistic, matrixLeft, matrixRight


def main():
    PATH1 = '/home/d3m1ur60/Desktop/LSBv2/Ayuda/ayuda_V1-0002.mp4'
    PATH_VIDEO = 'C:\\Users\\DougC\\Desktop\\LSBv1\\Ayuda\\ayuda_V1-0001.mp4'
    cap = cv2.VideoCapture(PATH_VIDEO)
    pastTime = 0
    detector = HolisticDetector()
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call Detection
        detector.find_pose(frame)  # Must be called for each detection
        detector.get_lm()
        detector.draw_pose(frame)  # Optional, just for drawing
        mat1, mat2, mat3 = detector.get_matrix()  # Returns holistic,left hand,right hand
        get_distMat(mat1, mat2, mat3)
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


if __name__ == '__main__':
    main()
