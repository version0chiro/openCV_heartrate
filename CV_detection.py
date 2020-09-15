import cv2
import imutils
import numpy as np
import time

print("[INFO] Start webcam")
time.sleep(1)
cap = cv2.VideoCapture(0)
while True:
    (grabbed,frame)=cap.read()
    if frame is None:
        break

    frame = imutils.resize(frame,width=600)
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    cv2.imshow("test",frame)
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


