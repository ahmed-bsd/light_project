import cv2
import numpy as np
from gtts import gTTS
import os
#import pygame

####  init #####
person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
light_cascade = cv2.CascadeClassifier('cascade2.xml')

# HSV red & green
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

cap = cv2.VideoCapture(0)
previous_color = None
person_detected = False

##### Functions  ######
def speaker(phrase):
    print(phrase)

def is_night():
    return False

##### Functions  ######

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Detect people in the image
        (rects, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(16, 16), scale=1.5)

        if len(rects)>0 and weights[0][0]>1:
            print("person detected ! ! ! ")
            print("rects" ,len(rects) , " weights : ", weights[0][0])
            # Draw rectangles around the detected people
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            print ("no person ..")

        lights = light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(lights) > 0 :
            founded_lights = sorted(lights, key=lambda x: x[0])

            for (x, y, w, h) in founded_lights:
                # Detect the color of the traffic light
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
                mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)

                if red_pixels>100 or green_pixels>100 :
                    print("red pixels : ",red_pixels)
                    print("green pixels : ", green_pixels)
                
                    if red_pixels > green_pixels:
                        color = 'red'
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        speaker("stop red")
                        
                    else:
                        color = 'green'
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        speaker("go , green")

        cv2.imshow('frame', frame)
        cv2.imshow('gray',gray)

        #Light level
        if is_night():
            # Add your own logic here to control LED
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
