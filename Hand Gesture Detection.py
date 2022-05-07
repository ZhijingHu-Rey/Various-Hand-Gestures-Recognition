# # Hand Gesture Recognizer 

import cv2
import csv
import copy
import math
import torch

# +
import numpy as np
import mediapipe as mp

import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
# -

from torchvision import transforms
from tensorflow.keras.models import load_model

path = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe"

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

transform=transforms.ToTensor()
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [max(0,x-20), max(0,y-20), min(x + w+20,image_width-1), min(y + h+20,image_height-1)]

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


# +
# Load the gesture recognizer model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(442, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 8)

    def forward(self, x, lm):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        print(np.shape(x))
        print(np.shape(lm))
        x = torch.cat((x, lm), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def predict(self, x, lm):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions except batch
        x = x.view(1,400)
        
        print(np.shape(x))
        print(np.shape(lm))
        
        x = torch.cat((x, lm), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

model = Net()
#model.load_state_dict(torch.load("./MyNet.pth")) # Model 1
model.load_state_dict(torch.load('DL_model.pt')) # Model 2 (Reponse faster, and have more accuracy)
model.eval()
# -

# Load class names
classNames = ["STOP","GOOD","YES","LOVE YOU","NO WAY","OKAY","GIMME A SECOND","CALL ME LATER"]

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            brect = calc_bounding_rect(debug_image, handslms)
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
            
            #print([landmarks])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            flatLM = torch.tensor(np.float32(np.reshape(landmarks, (1, -1))))

            cropped = framergb[brect[1]:brect[3], brect[0]:brect[2], :]
            reshaped = np.uint8(reshape(cropped, 28))
            
            # Predict gesture
            
            prediction = model.predict(transform(reshaped),torch.tensor(flatLM))
            #print(np.shape(torch.tensor(flatLM)))
            
            # Print(prediction)
            classID = np.argmax(prediction.detach().numpy())
            className = classNames[classID]

    # Show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
