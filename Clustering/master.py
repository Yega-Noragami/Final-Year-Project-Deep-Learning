#Import necessary libraries 

import os 

#Detection system 
def crop_using_detection():
    # python 1-detection/Crop_images_exif_multiple.py -m 1-detection/saved_model -l 1-detection/label_map.pbtxt -i images/ -o Results/crop/
    input_path = input("Enter image path: ")
    output_path = input("Enter output directory: ")

    cmd = 'python 1-detection/Crop_images_exif_multiple.py -m 1-detection/saved_model -l 1-detection/label_map.pbtxt -i {} -o {}'.format(input_path , output_path)
    print(cmd)

    os.system(cmd)

# Change picture size
def change_pic_size():
    size = input ("Enter size of image ")
    input_path = input("Enter image path: ")
    output_path = input("Enter output directory: ")

    cmd ='python 2-crop/converter_equal_image.py -i {} -o {}'.format(input_path, output_path)
    # print(cmd)
    os.system(cmd)

# reduce data using tsne_algorithm 
def tsne_reduce():
    input_path = input("Enter image path: ")
    output_path = input("Enter output csv path: ")

    cmd= 'python 3-tsne/cli.py tsne \'{}\' \'{}\' --feature-cols all --unique-col A -r 2'.format(input_path , output_path)
    # print(cmd)
    os.system(cmd)

# Open jupyter notebook and complete the clustering 
def clustering_images():
    os.system('cd Reults/')
    cmd='jupyter notebook'
    os.system(cmd)

# an easy fxn which will do everything for you 
def do_everything():
    # python 1-detection/Crop_images_exif_multiple.py -m 1-detection/saved_model -l 1-detection/label_map.pbtxt -i images/ -o Results/crop/
    cmd1='python 1-detection/Crop_images_exif_multiple.py -m 1-detection/saved_model -l 1-detection/label_map.pbtxt -i images/ -o Results/crop/'
    cmd2='python 2-crop/converter_equal_image.py -i Results/crop -o Results/square'
    cmd3='python 3-tsne/cli.py tsne \'./Results/square\' \'./Results/tsne-reduced.csv\' --feature-cols all --unique-col A'
    cmd4='cd 4-kmeans'
    cmd5='jupyter notebook'
    # os.system(cmd1)
    # os.system(cmd2)
    os.system(cmd3)
    # os.system(cmd4)
    # os.system(cmd5)
    print("Program execution completed!!!")


def main():
    while True:
        print('1. Carry detection and crop the input images ')
        print('2. Change all pictures to same size ')
        print('3. perform T-sne reduction ')
        print('4. Cluster all similar images into folder ')
        print('5. Do everything')
        print('6. Exit')

        choice = input("enter your choice: ")
        if choice == '6':
            print('Exiting !')
            break
        else:
            print('your choice is :', choice)

            if choice =='1':
                crop_using_detection()
            
            elif choice =='2':
                change_pic_size()

            elif choice =='3':
                tsne_reduce()

            elif choice =='4':
                clustering_images()

            elif choice =='5':
                do_everything()

if __name__ == "__main__":
    main()