import cv2
import mediapipe as mp

class hand_detector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection(self.min_detection_confidence, self.model_selection)
        self.mp_draw = mp.solutions.drawing_utils


    def find_hand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmark, self.mp_hands.HAND_CONNECTIONS)

        return img


    def get_landmark(self, img, draw=True):

        cords = []

        if self.results.hand_landmarks:
            for id, lm in enumerate(self.results.hand_landmarks.landmark):
                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)
                cords.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return cords


def main():
    cap = cv2.VideoCapture(0)
    detector = hand_detector()

    while True:
        success, img = cap.read()
        img = detector.find_hand(img)

        cv2.imshow("Hand Detector", img)

        cv2.waitKey(1)

if __name__ == "__main__":
    main()