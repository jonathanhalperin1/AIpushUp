import cv2
import mediapipe as md
import streamlit as st
import numpy as np
from PIL import Image

st.title("AI Computer Vision PushUp Detector")


drawing = md.solutions.drawing_utils
style = md.solutions.drawing_styles
Mpose = md.solutions.pose
count = 0
position = None

cap = cv2.VideoCapture(0)
frame = st.empty()
with Mpose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ListOfPose = []
        success, ret = cap.read()
        if not success:
            print("Webcam isn't Valid")
            break
        key = cv2.waitKey(1)

        ret = cv2.cvtColor(cv2.flip(ret, 1), cv2.COLOR_BGR2RGB)
        frame.image(ret,channels='RGB')
        result = pose.process(ret)

        if result.pose_landmarks:
            drawing.draw_landmarks(ret, result.pose_landmarks, Mpose.POSE_CONNECTIONS)
            for id,im in enumerate(result.pose_landmarks.landmark):
                h, w, _= ret.shape
                X,Y = int(im.x*w),int(im.y*h)
                ListOfPose.append([id,X,Y])

        if len(ListOfPose) != 0:
            if ((ListOfPose[12][2] - ListOfPose[11][2]) >= 15 and (ListOfPose[11][2] - ListOfPose[13][2]) >= 15):
                position = "down"
            if ((ListOfPose[12][2] - ListOfPose[14][2]) <= 5 and (
                    ListOfPose[11][2] - ListOfPose[13][2]) <= 5) and position == "down":
                position = "up"
                count += 1
                print(position)
                print(count)

            cv2.rectangle(ret, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(ret, str(position), (15, 12),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, cv2.LINE_AA)

            cv2.putText(ret, str(count), (10, 60),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)


            #cv2.imshow("Counter", cv2.flip(ret, 1))
            frame.image(ret, channels='RGB')

            if key==ord('e'):
                break

cap.release()
