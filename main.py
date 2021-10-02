import cv2
from modules.pose_tracker import pose_detector
from modules.hand_tracker import hand_detector

cap = cv2.VideoCapture(0)
detector_pose = pose_detector()
detector_hand = hand_detector()

while True:
    success, img = cap.read()

    detector_pose.find_pose(img)
    detector_hand.find_hand(img)

    cv2.imshow("Human Detector", img)

    cv2.waitKey(1)