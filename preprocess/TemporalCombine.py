import numpy as np
import cv2
import time
import json
import pickle

from os import listdir
from os.path import isfile, join

DEBUG = False

dataset0 = []
dataset1 = []
directory = '../dataset/processed/'
jsonName = 'onehot.json'
pickleName = 'dataset_temporal.pickle'
FPS = 20

with open(directory+jsonName) as fp:
    outputLabels = json.load(fp)

fileNames = [f for f in listdir(directory) if isfile(join(directory, f)) and \
             (join(directory, f).find(".avi") > 0)]
frames = None

for fileName in fileNames:
    cap = cv2.VideoCapture(directory+fileName)
    if fileName not in outputLabels:
        continue
    print(fileName)

    ret, frame = cap.read()
    width,height = frame.shape[1],frame.shape[0]

    oneHot = outputLabels[fileName]
    frameCounter = 0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newFileName = fileName.split('.')[0] + '_T.avi'
    out = cv2.VideoWriter(directory+newFileName,fourcc, FPS, (width,height))
    
    while cap.isOpened():
        frameCounter += 1   
        ret, frame = cap.read()

        # Break if video has ended
        if ret == False:
            break

        # Convert frame to black and white
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert to 3D array
        frame = frame.reshape((frame.shape[0],frame.shape[1],1))
        
        # Initialize array
        if frames == None:            
            frames = frame
            continue
        
        # Append channel
        frames = np.concatenate((frames, frame), axis=2)

        # Wait till the frame buffer is full    
        if frameCounter < 4:
            continue

        # Pop the oldest frame
        frames = frames[:,:,1:4]

        # Append to right dataset
        if frameCounter < len(oneHot):
            if oneHot[frameCounter][0] == 1:
                dataset0.append(frames)
            else:
                dataset1.append(frames)
        else:
            dataset1.append(frames)

        out.write(frames)
        
        # Display
        if DEBUG:
            if oneHot[frameCounter-1][0] == 1: # True
                cv2.imshow('win',frames)
            else: # Show the image darker
                cv2.imshow('win',(frames*.25).astype(np.uint8))
            if cv2.waitKey(int(1000/20)) & 0xFF == ord('q'):
                break
        
        
    cap.release()

cv2.destroyAllWindows()

minLength = min(len(dataset0),len(dataset1))

dataset = {'x':[[],[]]}
d = dataset['x']

np.random.shuffle(dataset0)
np.random.shuffle(dataset1)

for i in range(minLength):
    d[0].append(dataset0[i])
    d[1].append([1,0])

for i in range(minLength):
    d[0].append(dataset1[i])
    d[1].append([0,1])

#with open(directory+pickleName,'wb') as fp:
#    pickle.dump(dataset,fp)
    
