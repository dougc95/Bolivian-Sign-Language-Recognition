import cv2
import time
import numpy as np
import mediapipe as mp
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon
        # Create mp object,call pose object and drawing object
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody,self.smooth, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame, draw=True):
        # Model treats imgs as RGB
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # Draw detections
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, self.mpPose.UPPER_BODY_POSE_CONNECTIONS)
        return frame

    def getPosition(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 20, (255, 255, 0), 2)
        return self.lmList

    def getAngle(self, frame, p1, p2, p3, draw=True):
        # Get points
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]
        _, x3, y3 = self.lmList[p3]
        # Get angle
        angle = abs(math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                 math.atan2(y1 - y2, x1 - x2)))
        print(angle)
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 1)
            cv2.circle(frame, (x1, y1), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 0), -1)
            cv2.circle(frame, (x3, y3), 10, (255, 0, 0), -1)
        return angle