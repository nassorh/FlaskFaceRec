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

        self.camera = cv2.VideoCapture(0)
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
                #Half the size
                frame_resize = cv2.resize(frame, None, fx=self.SCALE_FACTOR, fy=self.SCALE_FACTOR)
                current_time = time.time()
                # Check if enough time has elapsed since the last face recognition
                if current_time - self.last_recognition_time >= 5:
                    print("Running fac rec..")
                    # Find all the faces and their encodings in the current frame
                    face_locations = face_recognition.face_locations(frame_resize)
                    face_encodings = face_recognition.face_encodings(frame_resize, face_locations)

                    # Iterate over each detected face
                    for (i, face_encoding) in enumerate(face_encodings):
                        # Determine if the face is a match for any known face

                        matches = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.6)

                        if len(self.known_faces) == 0:  # Check if there are no known faces
                            name = "Unknown"
                        else:
                            # Determine the best match for the detected face
                            best_match_idx = np.argmin(face_recognition.face_distance(self.known_faces, face_encoding))
                            if matches[best_match_idx]:
                                name = self.known_names[best_match_idx] 
                            else:
                                name = "Unknown"
                        
                        print(name)
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
def gen_camera():
    while fr_thread.keep_running:
        frame = fr_thread.frame

        # Reducing size of streaming
        frame = cv2.resize(frame, (640, 280))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')