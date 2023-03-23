from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os
from camera_test import camera
from face_rec import fr_thread,gen_camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ct')
def camera_test():
    return render_template('camera.html')
    
@app.route('/camera_test')
def plain_video():
    return Response(camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port="5001")



    










