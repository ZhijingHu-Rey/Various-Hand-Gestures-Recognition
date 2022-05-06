# +
import os
import torch

import cv2 as cv
import pandas as pd
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# -

import tensorflow as tf
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns

import csv
import copy
import math
import keras
import random
import warnings

# +
from PIL import Image as im
from keras.models import Sequential

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten,Dense,Dropout,MaxPool2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# +
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from random import randint

# +
classes = 8

landmarkpath = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\excel\landmarks_rand.csv"
imagepath = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\excel\images_rand.csv"
landmarktest = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\exceltest\landmarkstest_rand.csv"
imagetest = r"C:\Users\47637\Downloads\HW of DL\DL_Final\Hand Gesture Recognition Mediapipe\exceltest\imagestest_rand.csv"
# -

df_train = pd.read_csv(imagepath)
df_train.head()

df_test = pd.read_csv(imagetest)
df_test.head()

df_train.info()
df_train.shape

df_test.info()
df_test.shape

df_train["label"].unique()
number = df_train["label"].unique()

len(df_train["label"].unique())

sorted(df_train["label"].unique())

train_label = df_train["label"]
test_label = df_test["label"]
plt.style.use("ggplot")
plt.figure(figsize =(9,5))
sns.countplot(x= df_train['label'],data = df_train)
plt.show()

df_train.drop("label",axis=1,inplace=True)
df_train.head()

df_test.drop("label",axis=1,inplace=True)
df_test.head()

x_train = df_train.values
x_train

x_train = np.uint8(x_train.reshape(-1,28,28,3))
x_test = np.uint8(df_test.values.reshape(-1,28,28,3))

lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)

y_tr = np.argmax(y_train,axis=1)
y_tr

total_line = len(x_train)

# +
plt.figure(figsize=(9,7))

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i,:])
    plt.xlabel(y_tr[i])
    
plt.tight_layout()    
plt.show()

# +
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=(1./255),rotation_range = 30,
                                  width_shift_range = 0.2,height_shift_range =0.2,
                                  shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=(1./255))

# +
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# +
# Number of subprocesses to use for data loading
num_workers = 0
# How many samples per batch to load
batch_size = 20
# Percentage of training set to use as validation
valid_size = 0.2

# Convert data to a normalized torch.FloatTensor
transform =  transforms.Compose([
    transforms.ToTensor()
    ]) #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# +
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CustomImageDataset(Dataset):
    def __init__(self, imagepath, landmarkpath, transform=None):
        self.data = pd.read_csv(imagepath)
        self.landmarks=pd.read_csv(landmarkpath)
        
        self.label = self.landmarks["label"]
        #self.label = self.data["label"]
        
        self.data.drop("label",axis=1,inplace=True)
        self.landmarks.drop("label", axis=1, inplace=True)
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data.values[idx,:]
        x = np.uint8(np.reshape(x,(28,28,3)))
        lm = np.float32(self.landmarks.values[idx,:])
        label = self.label[idx]
        blabel = np.float32(np.zeros([classes]))
        blabel[label-1]=1
        blabel = torch.tensor(blabel)
        lm = torch.tensor(lm)
        if self.transform:
            x = self.transform(x)
        return x, blabel, lm, label


# -

trainset = CustomImageDataset(imagepath,landmarkpath,transform)
testset = CustomImageDataset(imagetest,landmarktest,transform)

# +
# Obtain training indices that will be used for validation
num_train = len(trainset)
num_test = len(testset)

# Method 1
train_set, valid_set = torch.utils.data.random_split(trainset, [int(0.8 * num_train), num_train - int(0.8 * num_train)])

# Method 2
# Define samplers for obtaining training, validation batches and test batches (Not neccessary)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=num_workers) #sampler=train_sampler,shuffle=True
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers) #sampler=valid_sampler,shuffle=True
test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=num_workers) 
# -

# Specify the image classes
Names = ['"Stop"', '"Good"', '"Yes"', '"Love you"', '"No way"',
           '"Okay"', '"Gimme a second"', '"Call me later"']
print(Names)


# +
# %matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


# +
import torch.nn as nn

# Model building
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
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = torch.cat((x, lm), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a complete CNN
model = Net().to(device)
print(model) # Network structure

# Move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

# +
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

# +
train_losslist = [] 
valid_losslist = []

valid_loss_min = np.Inf # Track change in validation loss

n_epochs = [*range(100)] # May increase this number to train a final model

for epoch in n_epochs:  # loop over the dataset multiple times

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    
    model.train()
    for i, data in enumerate(train_loader):
         
        inputs, labels, landmarks = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs, landmarks)
        # calculate the batch loss
        loss = criterion(output, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()
    
    ######################    
    # validate the model #
    ######################
    
    model.eval()
    for i, data in enumerate(valid_loader):
    
        inputs, labels, landmarks = data[0].to(device), data[1].to(device), data[2].to(device)
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(inputs, landmarks)
        # calculate the batch loss
        loss = criterion(output, labels)
        # update average validation loss 
        valid_loss += loss.item()
   
    # Calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_losslist.append(train_loss)
    valid_losslist.append(valid_loss)

    # Print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # Save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'DL_model.pt')
        valid_loss_min = valid_loss
# -


plt.plot(n_epochs, train_losslist, 'r', label='Training Loss')
plt.plot(n_epochs, valid_losslist, 'b', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Performance of the Current Model: Training and validation loss")
plt.legend()
plt.show()

# +
plt.figure(figsize=(12,8))

for i, data in enumerate(test_loader):
    
    #print(i)
    if(i>=10):
        break
    
    inputs, labels, landmarks, intlabel = data[0].to(device), data[1].to(device), data[2].to(device), data[3]

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(inputs, landmarks)
    # calculate the batch loss
    #loss = criterion(output, labels)
    # update average test loss 
    #test_loss += loss.item()
    
    npoutput = output.cpu().detach().numpy()
    npinput = inputs.detach().cpu().numpy()[0,:]
    
    npintlabel = intlabel.numpy()

    # Convert Tensor to image
    newimage = np.empty([28,28,3])
    
    for j in range(3):
        newimage[:,:,j] = npinput[j,:,:]
    
    #y_act = np.asscalar(np.array(npintlabel)) - 1  
    y_act = (np.array(npintlabel)).item() - 1
    y_pred = np.argmax(npoutput)

    plt.subplot(3,5,i+1)
    plt.imshow(newimage)
    plt.xlabel(f"Actual: {y_act}\n Predicted: {y_pred}")
    #print("done loop")

plt.tight_layout()
plt.show()

# average test loss
#test_loss = test_loss/len(test_loader.dataset)
# -


