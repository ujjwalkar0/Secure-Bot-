import numpy as np
import cv2

# face_cascade = cv2.CascadeClassifier_convert()

cap = cv2.VideoCapture(0)

while True:
    # Capture Frame-by-Frame
    ret,frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) and 0xFF == ord('q'):
        break

# When Everything done realese the capture
cap.release()
cv2.destroyAllWindows()
