import pytesseract
import cv2

video_capture = cv2.VideoCapture(0) # 0 indicates the default camera

config = '--psm 7' # Page segmentation mode for single character


def filter_contours(contours):
    min_width = 100
    min_height = 100
    max_width = 200
    max_height = 200

    filtered_contours = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        if w > min_width and h > min_height and w < max_width and h < max_height and aspect_ratio > 0.2 and aspect_ratio < 1.8:
            filtered_contours.append(contour)
    
    return filtered_contours

def recognize_letters(characters):
    recognized_letters = []
    for character in characters:
        recognized_letter = pytesseract.image_to_string(character, config=config)
        recognized_letters.append(recognized_letter)
    
    return recognized_letters
while True:
    # Capture a frame from the camera feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ######    letters detection   #############

    # Apply thresholding to the gray
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtring countours
    filtered_contours = filter_contours(contours)

    letters = []
    for contour in filtered_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        letter = gray[y:y+h, x:x+w]
        letters.append(letter)

    # OCR
    recognized_alpha = recognize_letters(letters)
    print("".join(recognized_alpha))
    # display detected letters on the frame
    for i, contour in enumerate(filtered_contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_letters[i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Alphabetic Letters', frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()