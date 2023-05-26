import cv2
import numpy as np

casc_classifier = cv2.CascadeClassifier('cascade2.xml')

# Define the lower and upper thresholds for red and green colors in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_green = np.array([36, 25, 25])
upper_green = np.array([86, 255, 255])

# Initialize the video capture object for the default camera
cap = cv2.VideoCapture(0)

# Initialize the previous color as None
prev_color = None

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect traffic lights in the grayscale frame
        traffic_lights = casc_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # If any traffic lights are detected
        if len(traffic_lights) > 0:
            # Sort the traffic lights by their x-coordinate (left to right)
            lights = sorted(traffic_lights, key=lambda x: x[0])

            # Loop through each traffic light
            for (x, y, w, h) in lights:
                # Extract the region of interest (ROI) around the traffic light
                roi = frame[y:y+h, x:x+w]

                # Convert the ROI to the HSV color space
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # Threshold the ROI to get only the red or green color regions
                mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
                mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

                # Count number of red and green pixels in the ROI
                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)

                # Determine the color based on the number of pixels
                if red_pixels > green_pixels:
                    color = 'Red'
                else:
                    color = 'Green'

                # If the color has changed from the previous frame
                if color != prev_color:
                    # Play a sound to indicate the change in color
                    if color == 'Red':
                        #playsound('stop.wav')
                        print("stop")
                    elif color == 'Green':
                        #playsound('go.wav')
                        print("go")
                    # Update the previous color
                    prev_color = color

                # Draw a rectangle around the traffic light and label it with its color
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255) if color == 'Red' else (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if color == 'Red' else (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
