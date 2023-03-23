import cv2
import numpy as np
import os

def camera():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
