# %%
wCam, hCam = 640, 480
frameR = 100
smoothening = 5

# %%

plocX, plocY = 0, 0
clocX, clocY = 0, 0

# %%
import cv2

# %%
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# %%
import mediapipe as mp

# %%
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

# %%
mpDraw = mp.solutions.drawing_utils
tipIds = [4, 8, 12, 16, 20]

# %%
# import autopy

# %%
# wScr, hScr = autopy.screen.size()
# print('Screen size:', wScr, hScr)

# %%
import math

# %%
def findHands(img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img, results

# %%
def findPosition(img, results, handNo=0, draw=True):
    lmList, bbox = [], []
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        xList, yList = [], []
        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax
        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
    return lmList, bbox

# %%
def fingersUp(lmList):
    fingers = []
    if len(lmList) == 0:
        return fingers
    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# %%
def findDistance(p1, p2, lmList, img, draw=True, r=15, t=3):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    length = math.hypot(x2 - x1, y2 - y1)
    return length, img, [x1, y1, x2, y2, cx, cy]

# %%

import numpy as np

# %%

# while True:
#     success, img = cap.read()
#     if not success:
#         break
#     img, results = findHands(img)
#     lmList, bbox = findPosition(img, results)
#     if len(lmList) != 0:
#         x1, y1 = lmList[8][1:]
#         x2, y2 = lmList[12][1:]
#         fingers = fingersUp(lmList)
#         cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
#         if fingers[1] == 1 and fingers[2] == 0:
#             x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
#             y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
#             clocX = plocX + (x3 - plocX) / smoothening
#             clocY = plocY + (y3 - plocY) / smoothening
#             autopy.mouse.move(wScr - clocX, clocY)
#             cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
#             plocX, plocY = clocX, clocY
#         if fingers[1] == 1 and fingers[2] == 1:
#             length, img, lineInfo = findDistance(8, 12, lmList, img)
#             if length < 40:
#                 cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
#                 autopy.mouse.click()
#     cv2.imshow('Virtual Mouse', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# %%
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer

# %%
st.set_page_config(page_title="AI Virtual Mouse", layout="centered")
st.title("AI Virtual Mouse")
st.write("Allow webcam access when prompted")
def process_frame(frame):
    try:
        global plocX, plocY, clocX, clocY
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        img, results = findHands(img)
        lmList, _ = findPosition(img, results)
        if len(lmList) > 12:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = fingersUp(lmList)
            cv2.rectangle(img, (frameR, frameR),(wCam - frameR, hCam - frameR),(255, 0, 255), 2)
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wCam))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hCam))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                cv2.circle(img, (int(clocX), int(clocY)), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY
            elif (fingers[1] == 1 and fingers[2] == 1) or (fingers[1] == 0 and fingers[2] == 1):
                length, img, lineInfo = findDistance(8, 12, lmList, img)
                if length < 40:
                    cx, cy = lineInfo[4], lineInfo[5]
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "CLICK", (cx - 40, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        print("Frame error:", e)
        return frame
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN
        {
            "urls": ["turn:relay1.expressturn.com:3478"],
            "username": "ef6E2qvB0a6Ft6pZxH9y6X5j8uWwNR",
            "credential": "WzA3Rr5c6Jd8nPsD2vYw3sFq1eTk8U"
        }
    ]
}
webrtc_streamer(key="virtual-mouse",video_frame_callback=process_frame,async_processing=True,rtc_configuration=RTC_CONFIGURATION,)