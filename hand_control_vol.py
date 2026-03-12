import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print('Camera initialized')

# Mediapipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Volume control setup for macOS
volMin, volMax = 0, 100

while True:
    success, img = cap.read()
    print(f'Frame success: {success}')
    if not success:
        continue
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    print(f'Hands detected: {num_hands}')
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            lmList = []

            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) != 0:
                x1, y1 = lmList[4]   # thumb
                x2, y2 = lmList[8]   # index finger

                cx, cy = (x1+x2)//2, (y1+y2)//2

                cv2.circle(img, (x1, y1), 10, (255,0,255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255,0,255), cv2.FILLED)

                cv2.line(img, (x1,y1),(x2,y2),(255,0,255),3)
                cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)

                length = math.hypot(x2-x1, y2-y1)
                print(f'  Finger distance: {length:.1f}px')

                # Convert finger distance to volume (macOS 0-100)
                vol = int(np.interp(length, [25, 180], [volMin, volMax]))
                print(f'  Setting volume: {vol}%')
                subprocess.run(['osascript', '-e', f'set volume output volume {vol}'], capture_output=True)

                # Volume bar
                volBar = np.interp(length, [25,180], [400,150])
                volPer = np.interp(length, [25,180], [volMin,volMax])

                cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
                cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
                cv2.putText(img,f'{int(volPer)} %',(40,450),
                            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

