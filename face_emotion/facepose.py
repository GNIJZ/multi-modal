import threading

import keyboard
# import win32api
# from pynput.mouse import Controller
# from Screen import grab_screen
import cv2
import mediapipe as mp
import time



# 将鼠标移动到指定位置（x, y）
# mouse.position = (2190, 1230)

mpPose=mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=0,
                   smooth_landmarks=True,
                   min_detection_confidence=0.7,
                   min_tracking_confidence=0.70
                   )

# mpFaceDetection = mp.solutions.face_detection
# face_detection = mpFaceDetection.FaceDetection(min_detection_confidence=0.5)
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)


class MediapipeThread(threading.Thread):
    def __init__(self,face_mesh):
        super().__init__()
        self.face_mesh=face_mesh
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()  # Read a frame from the webcam
            # img = grab_screen()
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            # Display the frame
            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            time.sleep(0.001)










