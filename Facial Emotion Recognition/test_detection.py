import cv2
import numpy as np
from keras.models import load_model

model = load_model('model_file.h5')
video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 3)
    for x,y,w,h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized_img = cv2.resize(sub_face_img,(48,48))
        normalize = resized_img/255.0
        reshaped = np.reshape(normalize, (1,48,48,1))
        result = model.predict(reshaped)
        lable = np.argmax(result, axis=1)[0]
        print(lable)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w, y), (50,50,255), -1)
        cv2.putText(frame, labels_dict[lable], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()