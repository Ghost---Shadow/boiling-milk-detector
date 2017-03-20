import numpy as np
import cv2

from os import listdir
from os.path import isfile, join

FPS = 20
SKIP = 5
INPUT_DIRECTORY = '../dataset/unprocessed/'
OUTPUT_DIRECTORY = '../dataset/processed/'

fileNames = [f for f in listdir(INPUT_DIRECTORY) if isfile(join(INPUT_DIRECTORY, f))]

for fileName in fileNames:
    print(fileName)
    cap = cv2.VideoCapture(INPUT_DIRECTORY+fileName)
    ret, frame = cap.read()
    width,height = frame.shape[1],frame.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    newFileName = fileName.split('.')[0] + '.avi'
    out = cv2.VideoWriter(OUTPUT_DIRECTORY+newFileName,fourcc, FPS, (width,height),False)

    c = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if c == 0:
            out.write(gray)
        c = (c+1)%SKIP

    cap.release()
    out.release()
