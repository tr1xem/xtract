import cv2
import numpy as np

def para_detect(file_name):
    # Read the image
    img = cv2.imread(file_name)
    
    img = cv2.bitwise_not(img)    
    # Convert to grayscale
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY_INV)

    # Remove noisy portions using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 3))
    dilated = cv2.dilate(new_img, kernel, iterations=5)
    output = cv2.bitwise_not(dilated)

    # Find contours
    contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ais = []
    for contour in contours:
        # Get rectangle bounding contour
        x, y, w, h = cv2.boundingRect(contour)
        a = w * h
        ais.append(a)

    ais.sort(reverse=True)

    # Draw contours on the original image for visualization
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out small false positives
        if w * h not in ais[:2]:
            continue
        
        # Draw rectangle around detected paragraph areas
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Save the output image with detected paragraphs
    cv2.imwrite('output4.png', img)

# Example usage
file_name = "files/input/page29.png"
para_detect(file_name)
