import cv2
import os
import time
from PIL import Image
import numpy as np
import threading
import face_recognition

class FaceRecognitionThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.faceCascade = cv2.CascadeClassifier('website/haarcascade_frontalface_default.xml')
        self.IMAGES_DIR = 'known_faces'
        self.known_faces = []
        self.known_names = []
        self.SCALE_FACTOR = 0.5
        for filename in os.listdir(self.IMAGES_DIR):
            image_path = os.path.join(self.IMAGES_DIR, filename)
            if self.check_image(image_path):
                image = cv2.imread(image_path)
                face_encoding = face_recognition.face_encodings(image)
                if len(face_encoding) > 0:  # Check if a face is found in the image
                    self.known_names.append(filename.split(".")[0])
                    self.known_faces.append(face_encoding[0])  # Add the encoding of the first found face in the image

        self.camera = cv2.VideoCapture(1)
        self.keep_running = True
        self.frame = None
        self.last_recognition_time = time.time()

    def check_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                return True
        except IOError:
            return False

    def run(self):
        while self.keep_running:
            success, frame = self.camera.read()
            if not success:
                break
            else:
                #Turn it gray and half the size
                gray = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
                current_time = time.time()
                # Check if enough time has elapsed since the last face recognition
                if current_time - self.last_recognition_time >= 5:
                    print("Running fac rec..")
                    faces = self.faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)
                    for (i, (x, y, w, h)) in enumerate(faces):
                        print("Found face...")
                        face = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, self.known_faces[0].shape[::-1])
                        
                        mse_list = []
                        for known_face in self.known_faces:
                            known_face_resized = cv2.resize(known_face, face_resized.shape[::-1])
                            diff = cv2.absdiff(known_face_resized, face_resized)
                            mse = np.mean(diff)
                            mse_list.append(mse)
                        best_match_idx = np.argmin(mse_list)

                        if mse_list[best_match_idx] < 60:
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # cv2.putText(frame, f'{self.known_names[best_match_idx]}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            print(self.known_names[best_match_idx])
                        else:
                            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            # cv2.putText(frame, 'No match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            print("Unknown")
                    self.last_recognition_time = current_time

                # Resize the frame to half of its original size
                # frame_resized = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
                # self.frame = frame_resized
                self.frame = frame

                # # Wait for 1 second before capturing the next frame
                time.sleep(0.01)

    def stop(self):
        self.keep_running = False
        self.join()

fr_thread = FaceRecognitionThread()
fr_thread.start()
