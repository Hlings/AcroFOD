## AcroFOD
The official PyTorch implementation of AcroFOD. The paper is accepted on the ECCV2022 and will be public soon.

### Things to do.
The commiter is fighting for refining the camera ready paper. So, there are some things to do later.
- Realease the processed ViPeD dataset if the original authors allow.
- Provide the sef of index or name for different target domain.
- Provide more training logs and pretrained checkpoints.
- Provide more tools and instructions for processing the raw data.

### Requirements and Preparing data
This repo is based on [YOLOv5 repo](https://github.com/ultralytics/yolov5). Please following that repo for installation and preparation.

### Training
1. Modify the config of data in the data subfoloder. Please refer the instruction in the yaml file.

2. The command below can reprouce the corresponding results mentioned in the paper.

```bash
python train_MMD.py --img 640 --batch 16 --epochs 300 --data ./data/city_and_foggy8_3.yaml --cfg ./models/yolov5x.yaml --hyp ./data/hyp_aug/m1.yaml --weights '' --name "test"
```
