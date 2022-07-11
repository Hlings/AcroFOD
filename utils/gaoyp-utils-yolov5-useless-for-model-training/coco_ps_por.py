import os
import shutil
import random

random.seed(12) # python coco_ps_por.py

proportion = 80 # 这里代表百分之多少 5 就是 5% 实际分数据集的时候就改这个就好了
# 已经完成的比例：1 5 10 30 40 50 60 70 80 90
cur_set = "train"

def filter_coco_person(proportion,cur_set):  
    in_path_txt = '/userhome/coco_person/labels' + '/' + cur_set
    in_path_img = '/userhome/coco_person/images' + '/' + cur_set

    out_path_img = "/userhome/coco_mini/coco_" + str(proportion) + "/" + "images" + '/' + cur_set
    out_path_txt = "/userhome/coco_mini/coco_" + str(proportion) + "/" + "labels" + '/' + cur_set

    txts = os.listdir(in_path_txt)
    txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']

    number_all = len(txts)

    number_filter = int((number_all*proportion)/100) # 到这里没有问题

    txts_filter = random.sample(txts,number_filter) # 根据对应的比例随机选择原始文档列表中的实例
    for every_txt in txts_filter: 
        number_cur = every_txt.split('.')[0] # 这一步是定义了根据数量筛选后的txt列表集合的第一个元素

        img_cur = number_cur + ".jpg"
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