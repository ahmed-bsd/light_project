import cv2
import numpy as np
from gtts import gTTS 
import os 

# Function of speaker
def speaker(phrase):
    language = 'en'
    output = gTTS(text=phrase, lang=language, slow=False)
    output.save("output.mp3")
    os.system("start output.mp3")


# person & traffic light cascade classifier
person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
light_cascade = cv2.CascadeClassifier('cascade2.xml')

# HSV red and green colors
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

# Initialization
cap = cv2.VideoCapture(0)
previous_color = None
person_detected = False

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect persons in the frame
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(persons) > 0:
            person_detected = True
            print("Person detected !")
            speaker("A person is in front of you !")

        else:
            person_detected = False

        # Detecting the traffic lights
        lights = light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(lights) > 0:
            founded_lights = sorted(lights, key=lambda x: x[0])

            for (x, y, w, h) in founded_lights:
                # Determine the color of the traffic light
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
                mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)

                if red_pixels > green_pixels:
                    color = 'Red'
                else:
                    color = 'Green'

                # notify the person by voice if a person is detected
                if color != previous_color and person_detected:
                    if color == 'Red':    
                        print("Stop")
                        speaker("Stop! The traffic light is red")
                    elif color == 'Green': 
                        print("Go")
                        speaker("You can go! The traffic light is green")
                    previous_color = color

                # Put a text on the traffic light
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255) if color == 'Red' else (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if color == 'Red' else (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
