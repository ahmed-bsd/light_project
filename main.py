import cv2
import numpy as np
from gtts import gTTS
import RPi.GPIO as GPIO
import time
import os


####  init #####
LED_PIN = 24
LDR_PIN = 23
MOTOR_PIN = 16

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(LDR_PIN, GPIO.IN)
GPIO.setup(MOTOR_PIN, GPIO.OUT)

person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
light_cascade = cv2.CascadeClassifier('cascade2.xml')

# HSV red & green
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

cap = cv2.VideoCapture(0)
previous_color = None
person_detected = False


####  init #####


##### Functions  ######
def speaker(phrase):
    language = 'en'
    output = gTTS(text=phrase, lang=language, slow=False)
    output.save("output.mp3")
    os.system("start output.mp3")

def is_night():
    ldr_value = GPIO.input(LDR_PIN)
    return ldr_value == GPIO.LOW

##### Functions  ######



while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(persons) > 0:
            person_detected = True
            print("Person detected !")
            speaker("A person is in front of you !")
            GPIO.output(MOTOR_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(MOTOR_PIN, GPIO.LOW)

        else:
            person_detected = False

        lights = light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if len(lights) > 0:
            founded_lights = sorted(lights, key=lambda x: x[0])

            for (x, y, w, h) in founded_lights:
                # Detect the color of the traffic light
                roi = frame[y:y+h, x:x+w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
                mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)

                if red_pixels > green_pixels:
                    color = 'red'
                else:
                    color = 'green'

                #voice notif
                if color != previous_color and person_detected:
                    if color == 'red':
                        print("Stop")
                        speaker("Stop! The traffic light is red")
                    elif color == 'green':
                        print("Go")
                        speaker("You can go! The traffic light is green")
                    previous_color = color
                #text on frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255) if color == 'red' else (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if color == 'red' else (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        #Light level
        if is_night():
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

GPIO.cleanup()
