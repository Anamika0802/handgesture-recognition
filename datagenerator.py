import cv2
import numpy as np
import os
import string
# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
if not os.path.exists("data/test"):
    os.makedirs("data/test")
for i in range(10):
    if not os.path.exists("data/train/" + str(i)):
        os.makedirs("data/train/"+str(i))
    if not os.path.exists("data/test/" + str(i)):
        os.makedirs("data/test/"+str(i))
if not os.path.exists("data/train/none"):
        os.makedirs("data/train/none")
if not os.path.exists("data/test/none"):
    os.makedirs("data/test/none")   


# Train or test 
mode = 'train' 
directory = 'data/'+mode+'/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1  

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Getting count of existing images
    count = {
             'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5")),
             'six': len(os.listdir(directory+"/6")),
             'seven': len(os.listdir(directory+"/7")),             
             'eight': len(os.listdir(directory+"/8")),             
             'nine': len(os.listdir(directory+"/9")),             
             'none': len(os.listdir(directory+"/none"))            

             }
    
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (10, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "ONE : "+str(count['one']), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "TWO : "+str(count['two']), (10, 130), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "THREE : "+str(count['three']), (10, 160), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "FOUR : "+str(count['four']), (10, 190), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "SIX : "+str(count['six']), (10, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "SEVEN : "+str(count['seven']), (10, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "EIGHT : "+str(count['eight']), (10, 310), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "NINE : "+str(count['nine']), (10, 340), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    cv2.putText(frame, "NONE : "+str(count['none']), (10, 370), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)
    
    # Drawing the ROI
    # print(frame.shape)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",gray)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    # th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    
    test_image = cv2.resize(test_image, (64,64))
    cv2.imshow("test", test_image)
        
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', test_image)       
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'6/'+str(count['six'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'7/'+str(count['seven'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'8/'+str(count['eight'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'9/'+str(count['nine'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'none/'+str(count['none'])+'.jpg', test_image)
   
    
cap.release()
cv2.destroyAllWindows()
