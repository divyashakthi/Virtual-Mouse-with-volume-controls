import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mpHands = mp.solutions.hands

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        results = self.hands.process(image)
        landMarkList = []

        if results.multi_hand_landmarks:  
            handrr = results.multi_hand_landmarks[handNumber] 
            for id, landMark in enumerate(handrr.landmark):
                imgH, imgW, imgC = originalImage.shape  
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])

            if draw:
                drawing_utils.draw_landmarks(originalImage, handrr, mpHands.HAND_CONNECTIONS)

        return landMarkList

handDetector = HandDetector(min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#print(volume.GetVolumeRange())

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    handLandmarks = handDetector.findHandLandMarks(image = frame, draw = True)
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand) 
            landmarks = hand.landmark
            #l_enm = enumerate(landmarks)
            #print(landmarks)

            for id,landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    cv2.circle(img=frame, center=(x,y), radius=15, color=(255, 0, 0))
                    index_x = screen_width/frame_width * x
                    index_y = screen_height/frame_height * y
                    pyautogui.moveTo(index_x, index_y)
                if id == 12:
                    cv2.circle(img=frame, center=(x,y), radius=15, color=(255, 0, 0))
                    middle_x = screen_width/frame_width * x
                    middle_y = screen_height/frame_height * y
                    #print("outside", abs(index_y - middle_y))
                    if abs(index_y - middle_y)<25:
                        pyautogui.click()
                        #pyautogui.sleep(1)
                '''if id ==4:
                    cv2.circle(img=frame, center=(x,y), radius=15, color=(255, 0, 0))
                    thumb_x = screen_width/frame_width * x
                    thumb_y = screen_height/frame_height * y'''
    if(len(handLandmarks)!=0):
        x1, y1 = handLandmarks[4][1], handLandmarks[4][2]
        x2, y2 = handLandmarks[8][1], handLandmarks[8][2]
        length = math.hypot(x2-x1, y2-y1)
                    #print(length)

        volumeValue = np.interp(length, [1,100], [-65.25, 0.0])
        volume.SetMasterVolumeLevel(volumeValue, None)

        cv2.circle(frame, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)