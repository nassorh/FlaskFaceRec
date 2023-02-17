from flask import Flask, render_template, Response
import cv2
import os
import numpy as np

IMAGES_DIR = 'known_faces'
FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EXIT_KEY = "q"

# Load the known face images and convert them to grayscale
known_faces = []
known_names = []
for filename in os.listdir(IMAGES_DIR):
    image_path = os.path.join(IMAGES_DIR, filename)

    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    known_names.append(filename.split(".")[0])
    known_faces.append(image_gray)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Iterate over each detected face
            for (i, (x, y, w, h)) in enumerate(faces):
                # Extract the face region from the frame and resize it to the same size as the known face images
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, known_faces[0].shape[::-1])

                # Compute the mean squared error (MSE) between the detected face and each known face image
                mse_list = []
                for known_face in known_faces:
                    diff = cv2.absdiff(known_face, face_resized)
                    mse = np.mean(diff)
                    mse_list.append(mse)

                # Determine the best match for the detected face
                best_match_idx = np.argmin(mse_list)
                if mse_list[best_match_idx] < 100:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{known_names[best_match_idx]}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, 'No match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
