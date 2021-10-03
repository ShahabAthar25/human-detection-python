import cv2
from modules.pose_detection import pose_detector
from modules.hand_detection import hand_detector

cap = cv2.VideoCapture(0)
detector_pose = pose_detector()
detector_hand = hand_detector()

while True:
    success, img = cap.read()

    detector_pose.find_pose(img)
    detector_hand.find_hand(img, draw=False)
    middle_fingle = detector_hand.get_landmark(img)
    print(middle_fingle[10:12])

    cv2.imshow("Human Detector", img)

    cv2.waitKey(1)