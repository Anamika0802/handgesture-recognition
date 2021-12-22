import numpy as np
from tensorflow.keras.models import model_from_json
import cv2
import sys,os
json_file = open("model-bw.json","r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("loaded......")

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    _,frame = cap.read()

    frame = cv2.flip(frame,1)

  # Drawing the ROI
    # print(frame.shape)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi,(64,64))
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    # th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow("test",test_image)
    
    result = loaded_model.predict(test_image.reshape(1,64,64,1))
    res = np.argmax(result)
    pred = ['zero','one','two','three','four','five','six','seven','eight','nine','none']
    cv2.putText(frame,pred[res],(10,120),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    cv2.imshow("frame",frame)
    if cv2.waitKey(10) & 0xFF ==27:
        break
cap.release()
cv2.destroyAllWindows()