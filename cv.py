import cv2
import numpy as np
import os
import uuid

def crop_images(image_paths, output_folder, debug=False):
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        imgi = cv2.imread(image_path)

        if imgi is None:
            print(f"Error: Could not read the image {image_path}.")
            continue 

        img = cv2.bitwise_not(imgi)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if debug:
            cv2.imwrite(f'{output_folder}/grayscale_image.png', gray)
            cv2.imwrite(f'{output_folder}/blurred_image.png', blurred)
            cv2.imwrite(f'{output_folder}/thresholded_image.png', thresh1)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            print(f"No contours found in {image_path}.")
            continue

        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

        sens = 10

        def merge_boxes(boxes):
            if not boxes:
                return []
            boxes = sorted(boxes, key=lambda b: b[0])
            merged_boxes = []
            current_box = list(boxes[0])
            for box in boxes[1:]:
                if box[0] <= current_box[2] + sens:
                    current_box[0] = min(current_box[0], box[0])
                    current_box[1] = min(current_box[1], box[1])
                    current_box[2] = max(current_box[2], box[2])
                    current_box[3] = max(current_box[3], box[3])
                else:
                    merged_boxes.append(tuple(current_box))
                    current_box = list(box)
            merged_boxes.append(tuple(current_box))
            return merged_boxes

        merged_boxes = merge_boxes(boxes)

        for idx, (x1, y1, x2, y2) in enumerate(merged_boxes):
            cropped_image = img[y1:y2, x1:x2]
            
            # Add filename as watermark
            filename_text = os.path.basename(image_path)
            font_scale = 0.5
            font_color = (255, 255, 255)  # White color
            font_thickness = 1
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size to position it correctly
            text_size = cv2.getTextSize(filename_text, font_face, font_scale, font_thickness)[0]
            text_x = cropped_image.shape[1] - text_size[0] - 5
            text_y = cropped_image.shape[0] - 5
            
            # Put text on image
            cv2.putText(cropped_image, filename_text, (text_x, text_y), font_face,
                        font_scale, font_color, font_thickness)

            output_image_path = os.path.join(output_folder,
                                               f'cropped_image_{os.path.basename(image_path).split(".")[0]}_{idx + 1}.png')
            cv2.imwrite(output_image_path, cropped_image)

def arrange_images_on_a4(input_folder):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))  

    A4_WIDTH_PIXELS = int(210 * 300 / 25.4)  
    A4_HEIGHT_PIXELS = int(297 * 300 / 25.4) 
    
    scaling_factor = 0.5  # Adjust this value to change the size of images on A4
    a4_sheet_index = 0
    a4_sheet = np.ones((A4_HEIGHT_PIXELS, A4_WIDTH_PIXELS, 3), dtype=np.uint8) * 255
    
    # Initialize offsets for placement
    x_offset = 50
    y_offset = 50

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        cropped_image = cv2.imread(image_path)

        if cropped_image is None:
            print(f"Error: Could not read the image {image_path}. Skipping.")
            continue 

        original_height, original_width = cropped_image.shape[:2]

        # Calculate new dimensions using scaling factor
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        # Resize the image
        cropped_image_resized = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        h_crop_resized, w_crop_resized = cropped_image_resized.shape[:2]

        # Check if the next image fits within the A4 sheet
        if y_offset + h_crop_resized > A4_HEIGHT_PIXELS:
            y_offset = 50
            x_offset += w_crop_resized + 20

        if x_offset + w_crop_resized > A4_WIDTH_PIXELS:
            # Save the current A4 sheet and start a new one
            output_a4_path = f'./files/output/a4_sheet_{a4_sheet_index}.png'
            os.makedirs(os.path.dirname(output_a4_path), exist_ok=True)  
            cv2.imwrite(output_a4_path, a4_sheet)
            print(f"A4 sheet saved at {output_a4_path}")

            # Reset offsets and create a new A4 sheet
            a4_sheet_index += 1
            a4_sheet.fill(255)  # Reset to white background
            x_offset = 50
            y_offset = 50

        # Place the resized image on the A4 sheet
        a4_sheet[y_offset:y_offset + h_crop_resized, x_offset:x_offset + w_crop_resized] = cropped_image_resized
        
        y_offset += h_crop_resized + 20

    # Save any remaining images on the last A4 sheet
    if y_offset > 50 or x_offset > 50: 
        output_a4_path = f'./files/output/a4_sheet_{a4_sheet_index}.png'
        os.makedirs(os.path.dirname(output_a4_path), exist_ok=True)  
        cv2.imwrite(output_a4_path, a4_sheet)
        print(f"A4 sheet saved at {output_a4_path}")

# Example usage
input_folder_path = './files/input/'   
output_folder_path = './files/processing/'   

# Get all images from input folder
input_image_paths = [os.path.join(input_folder_path, img) for img in os.listdir(input_folder_path) if img.endswith('.png')]
print("Input Image Paths:", input_image_paths)

# Crop images from input paths and save them to output folder
crop_images(input_image_paths, output_folder_path) 

# Arrange cropped images into A4 sheets
arrange_images_on_a4(output_folder_path)
