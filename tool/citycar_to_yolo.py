# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
import re
from tqdm import tqdm
import shutil

"""
   Similar to cityscpaes_to_yolo.py.
   The only difference is only considering the car class.
"""

# python cityscapes_to_yolo.py
def convert_annotation(image_id, paths):
    global label_map

    def find_box(points):
        _x, _y = [float(pot[0]) for pot in points], [float(pot[1]) for pot in points]
        return min(_x), max(_x), min(_y), max(_y)

    def convert(size, bbox):
        center_x, center_y = (bbox[0] + bbox[1]) / 2.0 - 1, (bbox[2] + bbox[3]) / 2.0 - 1
        center_w, center_h = bbox[1] - bbox[0], bbox[3] - bbox[2]
        return center_x / size[0], center_y / size[1], center_w / size[0], center_h / size[1]

    final_label_path, final_output_path = paths
    label_json_url = final_label_path / f'{image_id}_gtFine_polygons.json'
    # 输出到 ：final_output_path / f'{image_id}_leftImg8bit.txt'

    load_dict = json.load(open(label_json_url, 'r')) 
    output_cache = []
    for obj in load_dict['objects']: 
        obj_label = obj['label'] 
        
        if obj_label in ['out of roi', 'ego vehicle']:
            continue

        if obj_label not in label_map.keys():
            #label_map[obj_label] = len(label_map.keys())  
            continue

        x, y, w, h = convert((load_dict['imgWidth'], load_dict['imgHeight']), find_box(obj['polygon']))
        output_cache.append(f'{label_map[obj_label]} {x} {y} {w} {h}\n')
    
    shutil_img_flag = 0
    if len(output_cache) != 0:     
        with open(final_output_path / f'{image_id}_leftImg8bit.txt', 'w') as label_f:
            label_f.writelines(output_cache)
            label_f.close()
        shutil_img_flag = 1
    
    return shutil_img_flag


def mkdir(url):
    if not os.path.exists(url):
        os.makedirs(url)


if __name__ == '__main__':
    #root_dir = Path(__file__).parent
    root_dir = Path("/userhome/1_xin/da_datasets/cityscapes")
    print("root_dir is", root_dir)
    
    image_dir = root_dir / 'leftImg8bit'
    label_dir = root_dir / 'gtFine'
    image_output_root_dir = root_dir / 'images'
    label_output_root_dir = root_dir / 'labels'
    print("label dir is", label_dir)

    label_map = {'car': 0}  #  eg. {'car': 0, 'person': 1}
    
    for _t_ in tqdm(os.listdir(image_dir)):  # _t_ as ['train', 'test' 'val;]
        type_files = []
        mkdir(image_output_root_dir / _t_), mkdir(label_output_root_dir / _t_)
        for cities_name in os.listdir(image_dir / _t_):
            _final_img_path = image_dir / _t_ / cities_name  # root_dir / leftImg8bit / test / berlin
            _final_label_path = label_dir / _t_ / cities_name  # root_dir / getfine / test / berlin

            # berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019_gtFine_polygons.json
            image_ids = list(map(lambda s: re.sub(r'_leftImg8bit\.png', '', s), os.listdir(_final_img_path)))
            # print(names[:0])  -> berlin_000000_000019

            for img_id in image_ids:
                img_flag = convert_annotation(img_id, [_final_label_path, label_output_root_dir / _t_])
                img_file = f'{img_id}_leftImg8bit.png'
                if img_flag:
                    print(_final_img_path / img_file, image_output_root_dir / _t_ / img_file)
                    shutil.copy(_final_img_path / img_file, image_output_root_dir / _t_ / img_file) 
                    type_files.append(f'images/{_t_}/{img_id}_leftImg8bit.png\n')

        with open(root_dir / f'yolo_{_t_}.txt', 'w') as f:
            f.writelines(type_files)

    with open(label_output_root_dir / 'classes.txt', 'w') as f:
        for k, v in label_map.items():
            f.write(f'{k}\n')
    print([k for k in label_map.keys()], len([k for k in label_map.keys()]))
