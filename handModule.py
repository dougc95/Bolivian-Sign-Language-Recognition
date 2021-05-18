import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detecCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detecCon = detecCon
        self.trackCon = trackCon
        # Mediapie
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detecCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHand(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLm, self.mpHands.HAND_CONNECTIONS)
        # if self.results.palm_detections:
        #     for palms in self.results.palm_detections:
        #         print(palms.location_data.relative_bounding_box)
        #         self.mpDraw.draw_detection(frame, palms)
        return frame

    def getPosition(self,frame,handNo=0,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame,(cx,cy),15,(255,255,0),-1)
        return lmList

def main():
    # PATH = 'C:\\Users\\DougC\\Desktop\\Mediapipe\\prueba_V1-0001.mp4'
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    # Time
    pTime = 0
    while True:
        ret, frame = cap.read()
        frame = detector.findHand(frame)
        lmList = detector.getPosition(frame,draw=False)
        if len(lmList) !=0:
            print(lmList[0])
        # Show FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("Original", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
