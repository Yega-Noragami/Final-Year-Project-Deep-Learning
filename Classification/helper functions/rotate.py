

import os 
from cv2 import cv2
from random import seed 
from random import random 

# img = cv2.imread('normal/DSC09406.JPG')
# img_flip = cv2.flip(img, 1)
# cv2.imwrite('normal/DSC09406_flip.JPG', img_flip)
seed=(1)
read_directory = 'train/one/'
write_directory = 'train/one_aug/'

for file in os.listdir(read_directory):
    print(read_directory+file)
    img = cv2.imread(read_directory+file)
    height, width = img.shape[:2]
    rotatation_matrix = cv2.getRotationMatrix2D((width/2 , height/2), random()*10 , .9)
    rotated_image = cv2.warpAffine(img , rotatation_matrix, (width, height))
    cv2.imwrite(write_directory+'rotated_v2_'+ file , rotated_image)



# #find . -name '.DS_Store' -type f -delete
