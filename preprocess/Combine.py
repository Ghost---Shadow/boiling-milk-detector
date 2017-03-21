import numpy as np
import cv2
import time
import json
import pickle

from os import listdir
from os.path import isfile, join

DEBUG = False

dataset = {}
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
    
    dataset[fileName] = [[],[]]
    oneHot = outputLabels[fileName]
    frameCounter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break

        # Image
        dataset[fileName][0].append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Label
        if frameCounter < len(oneHot): 
            dataset[fileName][1].append(oneHot[frameCounter])
        else:
            dataset[fileName][1].append([0,1]) # False

        if DEBUG:
            if dataset[fileName][1][-1][0] == 1: # True
                cv2.imshow('win',dataset[fileName][0][-1])
            else: # Show the image darker
                cv2.imshow('win',(dataset[fileName][0][-1]*.25).astype(np.uint8))
            if cv2.waitKey(int(1000/20)) & 0xFF == ord('q'):
                break
        
        frameCounter += 1    
    cap.release()

cv2.destroyAllWindows()

with open(directory+pickleName,'wb') as fp:
        pickle.dump(dataset,fp)
    
