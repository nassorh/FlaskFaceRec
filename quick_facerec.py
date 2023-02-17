import cv2
import numpy as np
import os

# Load the known face images and convert them to grayscale
known_faces_dir = 'known_faces'
known_faces = []
for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    known_faces.append(image_gray)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the Haar cascades for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Loop over frames from the camera
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

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
            cv2.putText(frame, 'Match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f'Person {best_match_idx + 1}', (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'No match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
