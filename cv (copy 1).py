import cv2
import numpy as np

# Read image from which text needs to be extracted
imgi = cv2.imread("new.png")

img = cv2.bitwise_not(imgi)

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)

# Creating a copy of the original image to draw rectangles on
im2 = img.copy()

# List to hold bounding boxes
boxes = []

# Looping through the identified contours to get bounding boxes
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append((x, y, x + w, y + h))  # Store as (startX, startY, endX, endY)
sens = 10
# Function to merge nearby boxes
def merge_boxes(boxes):
    if not boxes:
        return []

    # Sort boxes by their x-coordinate
    boxes = sorted(boxes, key=lambda b: b[0])
    
    merged_boxes = []
    current_box = list(boxes[0])  # Start with the first box

    for box in boxes[1:]:
        # Check if there is an overlap or they are close enough (you can adjust this threshold)
        if box[0] <= current_box[2] + sens:  # If box starts before current box ends + threshold
            # Merge by updating current_box's coordinates
            current_box[0] = min(current_box[0], box[0])  # Min x
            current_box[1] = min(current_box[1], box[1])  # Min y
            current_box[2] = max(current_box[2], box[2])  # Max x
            current_box[3] = max(current_box[3], box[3])  # Max y
        else:
            merged_boxes.append(tuple(current_box))  # Save the merged box
            current_box = list(box)  # Start a new current box

    merged_boxes.append(tuple(current_box))  # Add the last merged box
    return merged_boxes

# Merge nearby bounding boxes
merged_boxes = merge_boxes(boxes)

# Draw rectangles for merged boxes on the image
for (x1, y1, x2, y2) in merged_boxes:
    rect = cv2.rectangle(im2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR (optional)
    cropped = im2[y1:y2, x1:x2]
    
    # Uncomment these lines to perform OCR and save recognized text
    # text = pytesseract.image_to_string(cropped)
    # file.write(text)
    # file.write("\n")

# Save the modified image with rectangles drawn around detected text areas
cv2.imwrite('output_with_merged_rectangles.png', im2)

# Optionally display the result (for debugging)
cv2.imshow('Image with Merged Rectangles', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()
