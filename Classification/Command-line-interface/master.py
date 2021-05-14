#Import necessary libraries 

import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, MaxPool2D, Conv2D, Flatten, Dropout, BatchNormalization
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import collections
import csv
import numpy
import pandas as pd
import efficientnet.keras as efn
from collections import Counter
from keras.applications.vgg16 import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
warnings.simplefilter(action='ignore', category = FutureWarning)



def classify():
    # Get input and output path 
    input_path = input("Enter image path: ")
    # output_path = input("Enter output directory: ")
    try:
        os.mkdir('testing')
    except:
        print('testing already exists!')
    cmd1= 'mv '+input_path+' testing/'
    os.system(cmd1)
    # Ask the user to enter his model of choise, you can select your own mode if you want as well !
    model_number = input(" Select model you want to choose(1, 2, 3, etc.): \n1. Best Model \n2. Grade-1 Biased \n")

    model_path = 'results/efficientnetb7_v2_finetune_April_added_finetune.h5'

    if model_number == 1:
        model_path = 'results/efficientnetb7_v2_finetune_April_added_finetune.h5'
    elif model_number == 2:
        model_path = 'results/efficientnetb7_v2_finetune_oneFinetune.h5'

    # Load data 

    testing_path = 'testing'
    # testing_path = input_path add  ------ fix folder issue 
    testing_batches = ImageDataGenerator(rescale = 1.0/255.).flow_from_directory(directory=testing_path, target_size=(224,224),batch_size=10,shuffle=False)
    categories = ['health', 'one', 'two', 'unused']

    # Load model 
    model = keras.models.load_model(model_path)

    # Predictions 
    predictions = model.predict(x=testing_batches, verbose=0)
   
    #Store predictions  
    list_predictions=list(np.argmax(predictions, axis=-1))

    # ----- Move images to respective folders 

    #filename
    list_filename = testing_batches.filenames
    list_unique=list(set(list_predictions))

    #create folders
    current = os.getcwd()
    for i in list_unique:
        os.mkdir(current +'/'+str(list_unique[i]))
    #     print(current+'/'+str(list_unique[i]))


    #get number of images in test
    length=len(list_filename)

    for i in range(length):
        shutil.copy('testing/'+list_filename[i],str(list_predictions[i]))
    #     print('testing/'+list_filename[i],str(list_predictions[i]))
    
    #rename files to correct Grade
    for i in list_unique:
        old = str(list_unique[i])
        new = str(categories[i])
        os.rename(old , new)
    
    # make results directory 
    try:
        os.mkdir('Classification')
    except:
        print("")

    for i in range(len(categories)):
        source= str(categories[i])
        targe='Classification'
        shutil.move(source, targe)
    


def main():

    while True:
        print('1. Classify images into Grade-1 , Grade-2, Health and Unused !')
 
        print('6. Exit')

        choice = input("enter your choice: ")
        if choice == '6':
            print('Exiting !')
            break
        else:
            print('your choice is :', choice)

            if choice =='1':
                classify()

if __name__ == "__main__":
    main()