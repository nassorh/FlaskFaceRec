from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os
from face_rec import fr_thread

def gen_camera():
    print(fr_thread.IMAGES_DIR)
    while fr_thread.keep_running:
        frame = fr_thread.frame

        # Reducing size of streaming
        frame = cv2.resize(frame, (640, 280))
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port="5001")



    










