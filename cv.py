import cv2
import numpy as np
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('image_processing.log')
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

def crop_images(image_paths, output_folder, debug=False):
    os.makedirs(output_folder, exist_ok=True)

    for image_path in image_paths:
        imgi = cv2.imread(image_path)
        if imgi is None:
            logging.error(f"Could not read the image {image_path}.")
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
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            logging.info(f"No contours found in {image_path}.")
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
            if cropped_image.size == 0:  
                logging.warning(f"Cropped image from {image_path} is empty.")
                continue

            filename_text = f"{os.path.basename(image_path).split('.')[0]} {idx + 1}.png"
            font_scale = 3.0
            font_color = (0, 0, 0)  
            font_thickness = 3
            font_face = cv2.FONT_HERSHEY_SIMPLEX

            text_size = cv2.getTextSize(filename_text, font_face, font_scale, font_thickness)[0]
            text_x = cropped_image.shape[1] - text_size[0] - 5
            text_y = cropped_image.shape[0] - 5

            cv2.putText(cropped_image, filename_text, (text_x, text_y), font_face,
                        font_scale, font_color, font_thickness)

            output_image_path = os.path.join(output_folder,
                                               f'cropped_image_{os.path.basename(image_path).split(".")[0]}_{idx + 1}.png')

            cv2.imwrite(output_image_path, cropped_image)
            logging.info(f"Saved cropped image: {output_image_path}")

def arrange_images_on_a4(input_folder):
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')],
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))

    A4_WIDTH_PIXELS = int(210 * 300 / 25.4)  
    A4_HEIGHT_PIXELS = int(297 * 300 / 25.4) 

    scaling_factor = 0.7 
    a4_sheet_index = 0
    a4_sheet = np.ones((A4_HEIGHT_PIXELS, A4_WIDTH_PIXELS, 3), dtype=np.uint8) * 255

    padding_x = 10 
    padding_y = 10 
    occupied_areas = [] 

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        cropped_image = cv2.imread(image_path)

        if cropped_image is None:
            logging.error(f"Could not read the image {image_path}. Skipping.")
            continue

        original_height, original_width = cropped_image.shape[:2]

        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)

        cropped_image_resized = cv2.resize(cropped_image,
                                            (new_width,
                                             new_height),
                                            interpolation=cv2.INTER_AREA)

        h_crop_resized, w_crop_resized= cropped_image_resized.shape[:2]

        placed = False

        for attempt_y in range(50, A4_HEIGHT_PIXELS - h_crop_resized - padding_y):
            for attempt_x in range(50, A4_WIDTH_PIXELS - w_crop_resized - padding_x):

                overlap_found = False

                for (ox1, oy1, ox2, oy2) in occupied_areas:
                    if not (attempt_x + w_crop_resized < ox1 or attempt_x > ox2 or 
                            attempt_y + h_crop_resized < oy1 or attempt_y > oy2):
                        overlap_found = True
                        break

                if not overlap_found:

                    a4_sheet[attempt_y:attempt_y + h_crop_resized,
                              attempt_x:attempt_x + w_crop_resized] = cropped_image_resized

                    occupied_areas.append((attempt_x,
                                            attempt_y,
                                            attempt_x + w_crop_resized,
                                            attempt_y + h_crop_resized))
                    placed = True
                    break

            if placed:
                break

        if not placed:
            output_a4_path=f'./files/output/a4_sheet_{a4_sheet_index}.png'
            os.makedirs(os.path.dirname(output_a4_path), exist_ok=True)

            cv2.imwrite(output_a4_path,a4_sheet)
            logging.info(f"A4 sheet saved at {output_a4_path}")

            a4_sheet_index += 1 
            a4_sheet.fill(255) 
            occupied_areas.clear() 

    output_a4_path=f'./files/output/a4_sheet_{a4_sheet_index}.png'
    os.makedirs(os.path.dirname(output_a4_path), exist_ok=True)

    cv2.imwrite(output_a4_path,a4_sheet) 
    logging.info(f"A4 sheet saved at {output_a4_path}")

input_folder_path='./files/input/' 
output_folder_path='./files/processing/'

input_image_paths=[os.path.join(input_folder_path,img) for img in os.listdir(input_folder_path) if img.endswith('.png')] 
logging.info("Input Image Paths: %s", input_image_paths)

crop_images(input_image_paths , output_folder_path)

arrange_images_on_a4(output_folder_path)
