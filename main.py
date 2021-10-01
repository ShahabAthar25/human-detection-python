import cv2
import modules.pose_tracker

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()



    cv2.imshow("Human Detector", img)

    cv2.waitKey(1)