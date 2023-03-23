from flask import Flask, render_template, Response
import face_recognition
import cv2
import numpy as np
import os
from face_rec import fr_thread,gen_camera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0",port="5001")



    










