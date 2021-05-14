import os
import shutil
import csv
import sys

current = os.getcwd()

with open('move_pictures.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    # line_count = 0
    # unique=[]
    # for row in csv_reader:
    #     if row[1] not in unique:    
    #         unique.append(row[1])

    # for i in range(len(unique)):
    #     os.mkdir(current+'/'+str(unique[i]))


    for row in csv_reader:
        # source = str(current+'/'+row[0])
        # destination=str(current+'/copy/'+row[1])
        os.rename(row[0],'copy/'+row[1])
        # print(source, destination)

