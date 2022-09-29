import cv2
import mediapipe as mp
import time

import facedetection as fd


cap = cv2.VideoCapture(0)
ptime=0
ctime=0
detector = fd.face_detection()
while True:
    status,img = cap.read()
    img,box=detector.findface(img)
    if len(box)!=0:
        print(box[0])
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    
    cv2.putText(img,f'FPS {str(int(fps))}',(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
