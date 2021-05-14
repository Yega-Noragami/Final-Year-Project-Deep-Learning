import os 
from cv2 import cv2

# img = cv2.imread('normal/DSC09406.JPG')
# img_flip = cv2.flip(img, 1)
# cv2.imwrite('normal/DSC09406_flip.JPG', img_flip)

read_directory = 'train/one/'
write_directory = 'train/one/'

for file in os.listdir(read_directory):
    print(read_directory+file)
    img = cv2.imread(read_directory+file)
    img_flip = cv2.flip(img , 1)
    print (write_directory+'flipped'+file)
    cv2.imwrite(write_directory+'flipped_'+file , img_flip)
