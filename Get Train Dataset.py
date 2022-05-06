# +
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import matplotlib.pyplot as plt

import csv
import copy
import cv2 as cv
import math
import argparse
import itertools

# +
from PIL import Image as im
from collections import Counter
from collections import deque

from model import KeyPointClassifier
from model import PointHistoryClassifier

from tensorflow.keras.models import load_model
from utils import CvFpsCalc


# -

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [max(0,x-20), max(0,y-20), min(x+w+20,image_width-1), min(y+h+20,image_height-1)]

def reshape(a,l):
    newimage=np.ndarray(shape=(l,l,3))
    hight=np.shape(a)[0]
    width=np.shape(a)[1]
    spaceh=math.floor(hight/l)
    spacew=math.floor(width/l)
    for k in range(3):
        for i in range(l):
            for j in range(l):
                newimage[i,j,k]=np.average(a[i*spaceh:i*spaceh+spaceh,j*spacew:j*spacew+spacew,k])
    return newimage

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info(image, fps):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    return image


landmarkpath = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\excel\landmarks.csv"
imagepath = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\excel\images.csv"

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

lmhandle = open(landmarkpath, 'a')
imghandle=open(imagepath, 'a')

# +
use_brect = True

# FPS
cvFpsCalc = CvFpsCalc(buffer_len=10)

while True:
    fps = cvFpsCalc.get()
    _, frame = cap.read()
    x, y, c = frame.shape
    frame = cv2.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    print(type(frame))
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, handslms)
            landmarks = []
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)


            flatten = np.reshape(landmarks,(1,-1))
            np.savetxt(lmhandle, flatten, delimiter=',')

            cropped = framergb[brect[1]:brect[3],brect[0]:brect[2],:]
            # print(np.shape(cropped))
            # plt.imshow(np.uint8(cropped),)
            # plt.show()
            reshaped=np.uint8(reshape(cropped,28))
            # plt.imshow(reshaped)
            # plt.show()
            # break
            img_flatten = np.reshape(reshaped,(1,-1))
            # print(np.shape(img_flatten))
            np.savetxt(imghandle, img_flatten, delimiter=',')
    
    frame = draw_info(frame, fps)
    
    cv2.imshow("Output", frame)
    # cv2.imshow("Output", cropped)
    
    if cv2.waitKey(1) == ord('q'):
        break

lmhandle.close()
imghandle.close()
# -

cap.release()
cv2.destroyAllWindows()

