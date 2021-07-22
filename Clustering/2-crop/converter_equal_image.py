#!/usr/bin/python
from PIL import Image
import os, sys
import argparse

global path 
global output
global size 

# path = "/home/noragami/Documents/21. CLUSTERING IMAGES/square-data-2/"


def resize():
    dirs = os.listdir(path)
    print("path yhi hai :",path)
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            fout, e = os.path.splitext(output+item)
            # to be deleted 
            print("this is the output path :", fout)
            try:
                imResize = im.resize((size,size), Image.ANTIALIAS)
                imResize.save(fout + '.jpg', 'JPEG', quality=100)
            except:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crop photots to your ratio')
    parser.add_argument("-i",'--image_path' , type = str , required = True , help="Input image folder")
    parser.add_argument('-s','--size', type = int , required=False , default = 512 , help="Enter Height X Width of the image")
    parser.add_argument('-o','--output_path', type = str , required=True , help='Enter output folder path')

    args = parser.parse_args()

    # os.mkdir(args.output_path)
    path= os.getcwd()+'/'+args.image_path+'/'

    size= args.size

    output = os.getcwd()+'/'+args.output_path+'/'

    resize()
