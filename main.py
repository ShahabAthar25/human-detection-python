import cv2
from modules.pose_tracker import pose_detector

cap = cv2.VideoCapture(0)
detector_pose = pose_detector()

while True:
    success, img = cap.read()

    detector_pose.find_pose(img)
    lm_list = detector_pose.get_specific_pose(img)
    cv2.circle(img, (lm_list[14][1], lm_list[14][2]), 15, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Human Detector", img)

    cv2.waitKey(1)