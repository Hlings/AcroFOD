from os import listdir
import os
import time
import ast
import sys
from PIL import Image

label_path = "/userhome/1_xin/da_datasets/KITTI/label_2/"
files = [f for f in listdir(label_path)]
output_path = "/userhome/1_xin/da_datasets/KITTI/labels/"
# os.mkdir(output_path)

for f in files:
    print(f) 
    label_file = label_path + f

    image_file_name = f.replace('.txt','.png')
    im = Image.open("/userhome/1_xin/da_datasets/KITTI/image_2/" + image_file_name)
    img_width, img_height = im.size

    c = 0
    count = 0
    for line in open(label_file, "r").readlines()[0:]: 
        l = line.split(" ")
        class_label = l[0]
        if class_label == "Car" : #or class_label == "Van" or class_label == "Truck" or class_label == "Tram" or class_label == "Pedestrian" or class_label == "Cyclist":
            count += 1

    for line in open(label_file, "r").readlines()[0:]:
        

        output_file = output_path + f
        output = open(output_file, "a+")

        dw = 1.0 / img_width
        dh = 1.0 / img_height

        l = line.split(" ")
        class_label = l[0]
        coords = list(map(float, l[4:8]))

        if class_label == "Car":
            class_label = 0
        else: 
            class_label = 6
        
        x = float((float(coords[2]) + float(coords[0])) / 2.0) * float(dw)
        y = float((float(coords[3]) + float(coords[1])) / 2.0) * float(dh)
        width = float(float(coords[2]) - float(coords[0])) * float(dw)
        height = float(float(coords[3]) - float(coords[1])) * float(dh)

        x = format(x, '.9f')
        y = format(y, '.9f')
        width = format(width, '.9f')
        height = format(height, '.9f')

        if class_label <= 0:
            output.write(str(class_label) + " " + x + " " + y + " " + width + " " + height)
            c += 1
            if c < count:
                output.write('\n')
        
    output.close()
"""
        elif class_label == "Van":
            class_label = 1
        elif class_label == "Truck":
            class_label = 2
        elif class_label == "Tram":
            class_label = 3
        elif class_label == "Pedestrian":
            class_label = 4
        elif class_label == "Cyclist":
            class_label = 5
 """
