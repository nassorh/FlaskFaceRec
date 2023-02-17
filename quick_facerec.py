import cv2
import numpy as np

# Load the known face image and convert it to grayscale
known = cv2.imread('hn103/website/Images/Hamad.png')
known_gray = cv2.cvtColor(known, cv2.COLOR_BGR2GRAY)

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
    for (x, y, w, h) in faces:
        # Extract the face region from the frame and resize it to the same size as the known face image
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, known_gray.shape[::-1])

        # Compute the absolute difference between the known face and the detected face
        diff = cv2.absdiff(known_gray, face_resized)

        # Compute the mean squared error (MSE) between the known face and the detected face
        mse = np.mean(diff)

        # If the MSE is below a certain threshold, the detected face is a match
        if mse < 100:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Match', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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

