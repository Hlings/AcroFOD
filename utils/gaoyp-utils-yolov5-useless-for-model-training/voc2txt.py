import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil

def convert(size, box):
    # size=(width, height)  b=(xmin, xmax, ymin, ymax)
    # x_center = (xmax+xmin)/2        y_center = (ymax+ymin)/2
    # x = x_center / width            y = y_center / height
    # w = (xmax-xmin) / width         h = (ymax-ymin) / height
    
    x_center = (box[0]+box[1])/2.0
    y_center = (box[2]+box[3])/2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    
    return (x,y,w,h)


def convert_annotation(xml_files_path, save_txt_files_path, imgs_in_path, imgs_out_path, classes):  
    
    img_files = os.listdir(imgs_in_path)
    img_files = [img for img in img_files if img.split('.')[-1] == 'jpg']
    
    xml_files = os.listdir(xml_files_path)
    xml_files = [xml for xml in xml_files if xml.split('.')[-1] == 'xml']
    
    for xml_name in xml_files:

        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        
        src_path = os.path.join(imgs_in_path, xml_name.split('.')[0] + '.jpg')
        dst_path = os.path.join(imgs_out_path, xml_name.split('.')[0] + '.jpg') 

        tree=ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            out_txt_f = open(out_txt_path, 'a')  # if obj's class not in class_list then no txt created               
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            #print(w, h, b)
            bb = convert((w,h), b)
            print("writng:",dst_path)
            shutil.copyfile(src_path, dst_path) #如果有class列表中的对象则将对应的图像复制出来
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":

    classes1 = ['car','bus','truck']
    #voc格式的xml标签文件路径
    xml_files1 = r'/userhome/coco_person/annotations/val'
    #转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = r'/userhome/coco_person/labels/val'
    #原始图片输入的路径
    imgs_in = r'/userhome/coco/val2017'
    imgs_out = r'/userhome/coco_person/images/val'
    
    convert_annotation(xml_files1, save_txt_files1, imgs_in, imgs_out,classes1)
