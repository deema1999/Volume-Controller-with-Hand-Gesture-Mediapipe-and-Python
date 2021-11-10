import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHand=2, detectionCo=0.5, trackingCo=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionCo = detectionCo
        self.trackinhCo = trackingCo

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.maxHand,
            self.detectionCo,
            self.trackinhCo)

    def findHands(self, img, draw=True):
        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False

        # Process the image and find hands
        self.results = self.hands.process(image)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        landmarkList = []
        #landmarkPositions = self.hands.process(img)
        landmarkCheck = self.results.multi_hand_landmarks
        if landmarkCheck:
            myHand = self.results.multi_hand_landmarks[handNo]
            for index, landmark in enumerate(myHand.landmark):
                # self.mp_drawing.draw_landmarks(
                #     img, hand, self.mp_hands.HAND_CONNECTIONS)
                h, w, c = img.shape
                centerX, centerY = int(landmark.x*w), int(landmark.y*h)
                landmarkList.append([index, centerX, centerY])
                if draw:
                    cv2.circle(img, (centerX, centerY), 7,
                               (255, 0, 255), cv2.FILLED)

        return landmarkList

    def main():
        cap = cv2.VideoCapture(0)
        detector = handDetector()

        while True:
            start = time.time()
            success, img = cap.read()
            img = detector.findHands(img)
            lmList = detector.findPosition(img)
            if len(lmList) != 0:
                print(lmList[4])

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            print("FPS: ", fps)

            cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Hands', img)

            cv2.waitKey(1)

    if __name__ == ("__main__"):
        main()
