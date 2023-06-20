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