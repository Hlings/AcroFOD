# -*- coding: utf-8 -*-
import os
import shutil
import random

random.seed(20) # python city_car_8_choice.py

"""
Instructions: 
1: You have prepared the yolo-style cityscapes dataset
2: change the in_path and out_path following
   "in_path" denotes the full dataset
   "out path" denotes the choiced subset
   
"""
proportion = 90
cur_set = "train"

def filter_coco_person(proportion,cur_set):  
    in_path_txt = '/userhome/1_xin/da_datasets/cityscapes_car/labels' + '/' + cur_set
    in_path_img = '/userhome/1_xin/da_datasets/cityscapes_car/images' + '/' + cur_set

    out_path_img = '/userhome/1_xin/da_datasets/cityscapes_car_8_2/images' + '/' + cur_set 
    out_path_txt = '/userhome/1_xin/da_datasets/cityscapes_car_8_2/labels' + '/' + cur_set 
    
    if not os.path.exists(out_path_img):
        os.makedirs(out_path_img)

    if not os.path.exists(out_path_txt):
        os.makedirs(out_path_txt)

    txts = os.listdir(in_path_txt)
    txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']

    number_all = len(txts)
    
    # you can define proportion or number for experiments
    #number_filter = int((number_all*proportion)/100)
    number_filter = 8

    txts_filter = random.sample(txts,number_filter)
    for every_txt in txts_filter: 
        number_cur = every_txt.split('.')[0]

        img_cur = number_cur + ".png"
        txt_cur = number_cur + ".txt"

        img_cur_in_path = in_path_img + "/" + img_cur
        img_cur_out_path = out_path_img + "/" + img_cur

        txt_cur_in_path = in_path_txt + "/" + txt_cur
        txt_cur_out_path = out_path_txt + "/" + txt_cur

        shutil.copy(img_cur_in_path,img_cur_out_path)
        shutil.copy(txt_cur_in_path,txt_cur_out_path)

        print(img_cur_out_path)
        print(txt_cur_out_path)
        print("next file")

filter_coco_person(proportion,cur_set)
print("all files are done")
