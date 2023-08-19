## (ECCV2022) AcroFOD: An Adaptive Method for Cross-domain Few-shot Object Detection

### Data Preparation
The links of the processed data (Yolo format) are as follows (in Baidu Desk):

[Sim10K](https://pan.baidu.com/s/1fd1hwyGkwn-cjBL5YPCAbg?pwd=juf6) Key: juf6 (The synthetic dataset includes only car class.)

[KITTI](https://pan.baidu.com/s/1edDtirk4IX9yFnsCGrzjDg?pwd=8brv) Key: 8brv (The KITTI dataset includes only car class.)

[Cityscapes_car_8_1](https://pan.baidu.com/s/1VjJn4aN5w9FdXzgIosr79Q?pwd=p69u) Key: p69u （The randomly selected 8 images from cityscapes_car.）

[Cityscapes_car](https://pan.baidu.com/s/1pU7NleGc-yG_JRLFjIKcxA?pwd=4ym4) Key: 4ym4 (The cityscapes dataset includes only car class.)

[Cityscapes_8cls](https://pan.baidu.com/s/1lPjaHOgoh5YCJcnP1hTzDw?pwd=rg4z) Key: rg4z (The Cityscapes dataset includes 8 classes.)

[Cityscapes_8cls_foggy](https://pan.baidu.com/s/1S1NuZSyailngL2M3STAZmA?pwd=y4yw) Key: y4yw (The Foggy Cityscapes dataset includes 9 classes.)

[Viped](https://pan.baidu.com/s/1a1SHZ4eb2q5mSyqWY2ZQmQ?pwd=a9y7) Key: a9y7 (The synthetic dataset includes)

[coco_person_60](https://pan.baidu.com/s/1VqpxNbjGjAMZvOF3HBttqw?pwd=vg1m) Key: vg1m (The randomly selected 60 images from coco_person.)

[coco_person](https://pan.baidu.com/s/1nwr7qVAFnXM3mK2b5Ywc9g?pwd=je89) Key: je89 (The COCO dataset includes only person class.)


You can also process the raw data to Yolo format via the tools shown [here](https://github.com/Hlings/AcroFOD/tree/main/tool).



### Requirements
This repo is based on [YOLOv5 repo](https://github.com/ultralytics/yolov5). Please follow that repo for installation and preparation.
The version I built for this project is YOLO v5 3.0. The proposed methods are also easy to be migrated into advanced YOLO versions.

### Training
1. Modify the config of the data in the data subfolders. Please refer to the instructions in the yaml file.

2. The command below can reproduce the corresponding results mentioned in the paper.

```bash
python train_MMD.py --img 640 --batch 12 --epochs 300 --data ./data/city_and_foggy8_3.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/m1.yaml --weights '' --name "test"
```

- If you find this paper/repository useful, please consider citing our paper:
```
@inproceedings{gao2022acrofod,
  title={AcroFOD: An Adaptive Method for Cross-Domain Few-Shot Object Detection},
  author={Gao, Yipeng and Yang, Lingxiao and Huang, Yunmu and Xie, Song and Li, Shiyong and Zheng, Wei-Shi},
  booktitle={European Conference on Computer Vision},
  pages={673--690},
  year={2022}
}
```
