# Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation) 
and [PoolFormer](https://github.com/sail-sg/poolformer) for the comparison, 


## Installation

Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMDetection v2.19.0](https://github.com/open-mmlab/mmdetection/tree/v2.19.0),
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full
mim install mmdet
```

## Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
detection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## ImageNet Pretraining
Put ImageNet-1K pretrained weights of backbone as 
```
EfficientFormer
├── weights
│   ├── efficientformer_l1_300d.pth
│   ├── ...
```

## Testing

Weights trained on COCO 2017 can be downloaded [here](https://drive.google.com/drive/folders/1eajQgA39bkPpyonzl8UnpiEwngVGaMdm?usp=sharing). 
We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
sh ./dist_test.sh config_file path/to/checkpoint #GPUs --eval bbox segm
```

For example, to test EfficientFormer-L1 on COCO 2017 on an 8-GPU machine, 

```
sh ./dist_test.sh configs/mask_rcnn_efficientformer_l1_fpn_1x_coco.py path/to/efficientformer_l1_coco.pth 8 --eval bbox segm
```

## Training
### Single machine multi-GPU training

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train EfficientFormer-L1 on an 8-GPU machine: 
```
sh ./dist_train.sh configs/mask_rcnn_efficientformer_l1_fpn_1x_coco.py 8
```
Tips: specify configs and #GPUs!

### Multi-node training
On Slurm-managed cluster, multi-node training can be launched by `slurm_train.sh`, similarly, to train EfficientFormer: 
```
sh ./slurm_train.sh your-partition exp-name config-file work-dir
```
Tips: specify GPUs/CPUs/memory per node in the script `slurm_train.sh` based on your resource!



