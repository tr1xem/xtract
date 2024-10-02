# import module
from pdf2image import convert_from_path
import cv2

# Store Pdf with convert_from_path function
images = convert_from_path('./files/test.pdf')

for i in range(29,34):
  
      # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.png', 'PNG')


# def cimage():
#
#     name = "new.png"
#
#
#       # Save pages as images in the pdf
#     images[33].save(name, 'PNG')
#
#
#
# cimage()

# opencv()
