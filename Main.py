import logging

from transformers import pipeline

from Whisperme import SpeechThread, AudioRecorder, Transcriber
from ThreadManager import ThreadManager
import keyboard
import cv2
import mediapipe as mp
import time
import os
from face_emotion import MediapipeThread
from Llm import chatspeak

if __name__ == '__main__':
    try:
        audio_recorder = AudioRecorder(channels=1, sample_rate=16000)
        transcriber = Transcriber(model_size="Whisperme/model/largev3/")
        classifier = pipeline("text-classification", model='Whisperme/model/bert_emotion', top_k=0)
        # speech_thread = SpeechThread(transcriber, audio_recorder,classifier,True)
        speech_thread = SpeechThread(transcriber, audio_recorder,True)
        # if message is not None:
        #     chatspeak(message,model="Llm/model/llama/",file="Llm/temp_audio.mp3")
        manager0 = ThreadManager(function=speech_thread)

        # 启动线程
        manager0.control_thread(True)

    except Exception as e:
        logging.error(e, exc_info=True)

    mpFaceMesh = mp.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(static_image_mode=False,
                                    max_num_faces=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
    mediapipe_thread=MediapipeThread(face_mesh)
    manager1 = ThreadManager(function=mediapipe_thread)
    manager1.control_thread(True)
