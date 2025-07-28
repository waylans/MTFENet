<div align="left">   


## MTFENet: A Multi-Task Autonomous Driving Network for Real-Time Target Perception
This repository(MTFENet) is the official PyTorch implementation of the paper "MTFENet: A Multi-Task Autonomous Driving Network for Real-Time Target Perception".  

---
### Update:
`2025-7-4`: We've uploaded the experiment results along with some code, and the full code will be released soon

### The Illustration of MTFENet

![MTFENet](pictures/MTFENet.pdf)


### Results
#### video visualization Results
* Note: The raw video comes from [HybridNets](https://github.com/datvuthanh/HybridNets/tree/main/demo/video/)
* The results of our experiments are as follows:
<td><img src=demo/2.gif/></td>

#### Evaluation of Model Size and Inference Speed.
| Networks       | Size (Pixel)| Parameters (M) | FPS(bs=32)  |
|----------------|-------------|----------------|-------------|
| YOLOP          | 640         | 7.90           | 361.27      |
| YOLOPv2        | 640         | 38.90          | 419.57      |
| YOLOPv3        | 640         | 30.2           | 389.58      |
| HybridNets     | 640         | 12.83          | 243.30      |
| A-YOLOM        | 640         | 13.61          | 346.53      |
| TriLiteNet     | 640         | **2.35**       | 423.50      |
| MTFENet        | 640         | 8.70           | **441.53**  |


#### Traffic Object Detection Result

| Model       | Recall (%) | mAP50 (%)  |
|-------------|------------|------------|
| Faster R-CNN| 81.20      | 64.90      |
| YOLOV5s     | 77.20      | 86.80      |
| MultiNet    | 81.30      | 60.20      |
| DLT-Net     | **89.40**  | 68.40      |
| YOLOP       | 76.50      | 88.20      |
| HybridNets  | 77.30      | 89.70      |
| YOLOPv2     | 83.40      | 91.10      |
| A-YOLOM     | 81.10      | 86.90      |
| YOLOPv3     | 84.30      | **96.90**  |
| TriLiteNet  | 72.30      | 85.60      |
| MTFENet     | 81.50      | 88.40      |

#### Drivable Area Segmentation Result

| Model          | mIoU (%) |
|----------------|----------|
| MultiNet       | 71.6     |
| DLT-Net        | 72.1     |
| PSPNet         | 89.6     |
| YOLOP          | 78.1     |
| YOLOP          | 91.6     |
| YOLOPv2        | 90.5     |
| A-YOLOM        | 91.0     |
| YOLOPv3        | 93.20    |
| TriLiteNet     | 92.40    |
| MTFENet        | **93.80**| 


#### Lane Detection Result:

| Model          | Accuracy (%) | IoU (%) |
|----------------|--------------|---------|
| Enet           | 34.12        | 14.64   |
| SCNN           | 35.79        | 15.84   |
| ENet-SAD       | 36.56        | 16.02   |
| YOLOP          | 84.40        | 26.50   |
| HybridNets     | 85.40        | 31.60   |
| A-YOLOM(s)     | 84.90        | 28.80   |
| YOLOPv3        | **88.30**    | 28.00   |
| TriLiteNet     | 82.30        | 29.80   |
| MTFENet        | 87.60        | 33.70   |
 

#### Ablation Studies 1: Adaptive concatenation module:

| Training method | Recall (%) | mAP50 (%) | mIoU (%) | Accuracy (%) | IoU (%) |
|-----------------|------------|-----------|----------|--------------|---------|
| YOLOM(n)        | 85.2       | 77.7      | 90.6     | 80.8         | 26.7    |
| A-YOLOM(n)      | 85.3       | 78        | 90.5     | 81.3         | 28.2    |
| YOLOM(s)        | 86.9       | 81.1      | 90.9     | 83.9         | 28.2    |
| A-YOLOM(s)      | 86.9       | 81.1      | 91       | 84.9         | 28.8    |


#### Ablation Studies 2: Results of different Multi-task model and segmentation structure:

| Model          | Parameters | mIoU (%) | Accuracy (%) | IoU (%) |
|----------------|------------|----------|--------------|---------|
| YOLOv8(segda)  | 1004275    | 78.1     | -            | -       |
| YOLOv8(segll)  | 1004275    | -        | 80.5         | 22.9    |
| YOLOv8(multi)  | 2008550    | 84.2     | 81.7         | 24.3    |
| YOLOM(n)       | 15880      | 90.6     | 80.8         | 26.7    |

YOLOv8(multi) and YOLOM(n) only display two segmentation head parameters in total. They indeed have three heads, we ignore the detection head parameters because this is an ablation study for segmentation structure.

---

### Visualization

#### Real Road

![Real Rold](pictures/real-road.png)

---

### Requirement
We implemented the algorithm in a Linux environment and conducted all experiments on an NVIDIA RTX 4090 GPU equipped with 24GB of memory.
The development environment was based on Python==3.8.19(https://www.python.org/) ,PyTorch 1.13.1(https://pytorch.org/get-started/locally/), conda 24.1.2, and CUDA11.7, with pre-training conducted on the BDD100K dataset. 

```setup
cd MTFENet
pip install -e .
```
### Dataset
- Download the images from [images](https://bdd-data.berkeley.edu/). 
- Download the annotations of detection from [detection-object](https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/EflGScMT-D1MqBTTYUSMdaEBT1wWm5uB8BausmS7fDLsQQ?e=cb7age). 
- Download the annotations of drivable area segmentation from [seg-drivable-10](https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/EWyIyXDFCzRLhERniUiuyIABq257WF4DbNJBDB8Dmok91w?e=hgWtoZ). 
- Download the annotations of lane line segmentation from [seg-lane-11](https://uwin365-my.sharepoint.com/:u:/g/personal/wang621_uwindsor_ca/EUBQBO2KGFtHsexik3WvLZMBuaW1CsnHDTZo5eJ3ESdJNA?e=K6Tsem). 

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train2017
│ │ ├─val2017
│ ├─detection-object
│ │ ├─labels
│ │ │ ├─train2017
│ │ │ ├─val2017
│ ├─seg-drivable-10
│ │ ├─labels
│ │ │ ├─train2017
│ │ │ ├─val2017
│ ├─seg-lane-11
│ │ ├─labels
│ │ │ ├─train2017
│ │ │ ├─val2017
```

Update the your dataset path in the `./test_yaml/bdd-mtfenet-multi.yaml`.

### Training

```
python ./ultralytics/train.py
```
### Evaluation

```
python ./ultralytics/val.py
```
### Prediction

```
python ./ultralytics/predict.py
```

  
**Notes**: 
We would like to express our sincere appreciation to the authors of the following works for their valuable contributions to the field of multi-task visual perception. Their research has provided strong foundations and meaningful benchmarks that have significantly guided and inspired our study. We also gratefully acknowledge the open-source code repositories they provided, which facilitated fair comparison and reproducibility in our experiments:

* **MultiNet** – [Paper](https://arxiv.org/pdf/1612.07695.pdf), [Code](https://github.com/MarvinTeichmann/MultiNet)
* **DLT-Net** – [Paper](https://ieeexplore.ieee.org/abstract/document/8937825)
* **Faster R-CNN** – [Paper](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf), [Code](https://github.com/ShaoqingRen/faster_rcnn)
* **YOLOv5s** – [Code](https://github.com/ultralytics/yolov5)
* **PSPNet** – [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf), [Code](https://github.com/hszhao/PSPNet)
* **ENet** – [Paper](https://arxiv.org/pdf/1606.02147.pdf), [Code](https://github.com/osmr/imgclsmob)
* **SCNN** – [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16802/16322), [Code](https://github.com/XingangPan/SCNN)
* **ENet-SAD** – [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Hou_Learning_Lightweight_Lane_Detection_CNNs_by_Self_Attention_Distillation_ICCV_2019_paper.pdf), [Code](https://github.com/cardwing/Codes-for-Lane-Detection)
* **YOLOP** – [Paper](https://link.springer.com/article/10.1007/s11633-022-1339-y), [Code](https://github.com/hustvl/YOLOP)
* **HybridNets** – [Paper](https://arxiv.org/abs/2203.09035), [Code](https://github.com/datvuthanh/HybridNets)
* **YOLOv8** – [Code](https://github.com/ultralytics/ultralytics)
* **A-YOLOM** – [Paper](https://arxiv.org/pdf/2310.01641.pdf), [Code](https://github.com/JiayuanWang-JW/YOLOv8-multi-task)
* **YOLOPv3** – [Paper](https://www.mdpi.com/2072-4292/16/10/1774), [Code](https://github.com/jiaoZ7688/YOLOPv3)
* **TriLiteNet** – [Paper](https://ieeexplore.ieee.org/document/10930421), [Code](https://github.com/chequanghuy/TriLiteNet)

