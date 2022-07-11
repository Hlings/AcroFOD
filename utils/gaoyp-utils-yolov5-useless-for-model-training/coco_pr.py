import os
import shutil
import random

 # python coco_ps_por.py

    
# 接下来要把所有YOLO格式的数据转化为COCO格式的数据

#proportion = 1 # 这里代表百分之多少 5 就是 5% 实际分数据集的时候就改这个就好了 1对应的第一个随机种子是12
# 已经完成的比例：1 5 10 30 40 50 60 70 80 90
proportion_set = [5,10,20,30,40,50,60,70,80,90]

def filter_data(proportion,cur_set,seed=12, cur_pos=1): #seed 代表随机种子  cur_pos 代表当前第几个随机
    random.seed(seed)  
    in_path_txt = '/userhome/coco_person/labels' + '/' + cur_set
    in_path_img = '/userhome/coco_person/images' + '/' + cur_set

    #out_path_img = "/userhome/voc_mini/voc_" + str(proportion) + "/" + "images" + '/' + cur_set
    out_path_img = "/userhome/coco_mini/coco_" + str(proportion) + "/" + "coco_" + str(proportion) +"_" + str(cur_pos) + "/" + "images" + '/' + cur_set
    if not os.path.exists(out_path_img):
        os.makedirs(out_path_img)
    #out_path_txt = "/userhome/voc_mini/voc_" + str(proportion) + "/" + "labels" + '/' + cur_set
    out_path_txt = "/userhome/coco_mini/coco_" + str(proportion) + "/" + "coco_" + str(proportion) +"_" + str(cur_pos) + "/" + "labels" + '/' + cur_set
    if not os.path.exists(out_path_txt):
        os.makedirs(out_path_txt)

    txts = os.listdir(in_path_txt)
    txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']

    number_all = len(txts)

    number_filter = int((number_all*proportion)/100) 

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


proportion_set = [1,5,10,20]
cur_pos = 1   #记得要改 代表当前是第几个实验
cur_set = "train"

for proportion in proportion_set:
    cur_pos = 1
    for seed in range(13,22): #当前是 seed 12-21 一共十个子数据集
        cur_pos += 1
        filter_data(proportion,cur_set,seed,cur_pos)
        
print("all files are done")