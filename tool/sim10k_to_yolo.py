# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#python sim10k_to_yolo.py
classes = ['car']
def gen_classes(image_id):
    in_file = open('%s/Annotations/%s.xml'%(path,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
    return classes

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 
    y = (box[2] + box[3])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_set,image_id):
    in_file = open('%s/Annotations/%s.xml'%(path,image_id))
    out_file = open('%s/%s'%(path,image_set)+'labels/%s.txt'%image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1: # only caring car class
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


path = '/userhome/1_xin/da_datasets/sim10k/VOC2012'

img_folder = os.path.join(path, "JPEGImages")
imgs_set = os.listdir(img_folder)
sets = []
for img in imgs_set:
    sets.append(img.split(".")[0]) # img : 3400909 

image_set = "train"

for image_id in sets:
    if not os.path.exists('%s/%s'%(path,image_set)+'labels/'):
        os.makedirs('%s/%s'%(path,image_set)+'labels/')
        
    #image_ids = open('%s/ImageSets/Main/%s.txt'%(path,image_set)).read().strip().split()
    #for image_id in image_ids:
    #gen_classes(image_id)
    convert_annotation(image_set,image_id)

classes_file = open(os.path.join(path,'classes.txt'),'w')
classes_file.write("\n".join([a for a in classes]))
classes_file.close()

print("all files done")
