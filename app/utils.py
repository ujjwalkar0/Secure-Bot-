import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    # Capture Frame-by-Frame
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,scaleFactor=1.5)

    for (x,y,w,h) in face:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h

        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)
        
    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) and 0xFF == ord('q'):
        break

# When Everything done realese the capture
cap.release()
cv2.destroyAllWindows()
