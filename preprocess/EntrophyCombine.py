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
pickleName = 'dataset.pickle'

with open(directory+jsonName) as fp:
    outputLabels = json.load(fp)

fileNames = [f for f in listdir(directory) if isfile(join(directory, f)) and \
             (join(directory, f).find(".avi") > 0)]

for fileName in fileNames:
    cap = cv2.VideoCapture(directory+fileName)
    if fileName not in outputLabels:
        continue
    print(fileName)

    oneHot = outputLabels[fileName]
    frameCounter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        if frameCounter < len(oneHot):
            if oneHot[frameCounter][0] == 1:
                dataset0.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            else:
                dataset1.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            dataset1.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if DEBUG:
            if oneHot[frameCounter][0] == 1: # True
                cv2.imshow('win',frame)
            else: # Show the image darker
                cv2.imshow('win',(frame*.25).astype(np.uint8))
            if cv2.waitKey(int(1000/20)) & 0xFF == ord('q'):
                break
        
        frameCounter += 1    
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

with open(directory+pickleName,'wb') as fp:
    pickle.dump(dataset,fp)
    
