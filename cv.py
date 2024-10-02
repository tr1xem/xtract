import cv2
import numpy as np
import os

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
            
           
            output_image_path = os.path.join(output_folder, f'cropped_image_{idx + 1}.png')
            cv2.imwrite(output_image_path, cropped_image)

def arrange_images_on_a4(input_folder):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))  

    A4_WIDTH_PIXELS = int(210 * 300 / 25.4)  
    A4_HEIGHT_PIXELS = int(297 * 300 / 25.4) 
    
    a4_sheet = np.ones((A4_HEIGHT_PIXELS, A4_WIDTH_PIXELS, 3), dtype=np.uint8) * 255
    x_offset = 50
    y_offset = 50

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        cropped_image = cv2.imread(image_path)

        original_height, original_width = cropped_image.shape[:2]
        
                scaling_factor = min((A4_WIDTH_PIXELS - x_offset - 20) / original_width,
                             (A4_HEIGHT_PIXELS - y_offset - 20) / original_height)

        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        cropped_image_resized = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        h_crop_resized, w_crop_resized = cropped_image_resized.shape[:2]

        if y_offset + h_crop_resized > A4_HEIGHT_PIXELS:
            y_offset = 50
            x_offset += w_crop_resized + 20

        if x_offset + w_crop_resized > A4_WIDTH_PIXELS:
            break

        a4_sheet[y_offset:y_offset + h_crop_resized, x_offset:x_offset + w_crop_resized] = cropped_image_resized
        
        y_offset += h_crop_resized + 20

    output_a4_path = './files/output/output_a4_sheet.png'
    
      cv2.imwrite(output_a4_path, a4_sheet)

    # for image_file in image_files:
    #     os.remove(os.path.join(input_folder, image_file))

# Example usage
input_image_paths = [
    "./files/input/new.png",
     "./files/input/page31.png",
    "./files/input/page32.png", 
    "./files/input/page33.png",  
]

output_folder_path = './files/processing/'   

crop_images(input_image_paths, output_folder_path) 


arrange_images_on_a4(output_folder_path)
