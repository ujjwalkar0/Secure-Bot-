import cv2
import os
import numpy as np
from PIL import Image

image_dir = './images'

face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_alt2.xml')

current_id = 0

label_ids = {}

y_labels = []
x_train = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ",".").lower()

            print(path,label)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            print(label_ids)

            # y_labels.append(label)
            # x_train.append(path)

            pil_image = Image.open(path)#.convert('L')
            image_array = np.array(pil_image,'uint8')
            print(image_array)

            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
            print("""
            --------------------------------------------
            """,faces)

print(y_labels)
print(x_train)