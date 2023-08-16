## AcroFOD: An Adaptive Method for Cross-domain Few-shot Object Detection
The official PyTorch implementation of AcroFOD. The paper is accepted on the ECCV2022 and will be public soon.

### Things to do.
The committee is fighting to refine the camera-ready paper. So, there are some things to do later.
- Release the processed ViPeD dataset if the original authors allow it.
- Provide the sef of index or name for different target domains.
- Provide more training logs and pretrained checkpoints.
- Provide more tools and instructions for processing the raw data.

**The above data has been uploaded. I will share the links this week.**

### Requirements and Preparing data
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
