# Semantic Segmentation 

Segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We follow the hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation) 
and [PoolFormer](https://github.com/sail-sg/poolformer) for the comparison. 

## Requirements
Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMSegmentation v0.19.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.19.0). 
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install -U openmim
mim install mmcv-full
mim install mmseg
```

## Data preparation

We benchmark EfficientFormer on the challenging ADE20K dataset, which can be downloaded and prepared following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```



## Testing

Weights trained on ADE20K can be downloaded [here](https://drive.google.com/drive/folders/1sklA_2Q8_2f-RaOYSj4n8rDGAsV-wh5g?usp=sharing). 
We provide a multi-GPU testing script, specify config file, checkpoint, and number of GPUs to use: 
```
sh ./tools/dist_test.sh config_file path/to/checkpoint #GPUs --eval mIoU
```

For example, to test EfficientFormer-L1 on ADE20K on an 8-GPU machine, 

```
sh ./tools/dist_test.sh configs/sem_fpn/fpn_efficientformer_l1_ade20k_40k.py path/to/efficientformer_l1_ade20k.pth 8 --eval mIoU
```

## Training 

### ImageNet Pretraining
Put ImageNet-1K pretrained weights of backbone as 
```
EfficientFormer
├── weights
│   ├── efficientformer_l1_300d.pth
│   ├── ...
```
### Single machine multi-GPU training

We provide PyTorch distributed data parallel (DDP) training script `dist_train.sh`, for example, to train EfficientFormer-L1 on an 8-GPU machine: 
```
sh ./tools/dist_train.sh configs/sem_fpn/fpn_efficientformer_l1_ade20k_40k.py 8
```
Tips: specify configs and #GPUs!

### Multi-node training
On Slurm-managed cluster, multi-node training can be launched by `slurm_train.sh`, similarly, to train EfficientFormer: 
```
sh ./tools/slurm_train.sh your-partition exp-name config-file 
```
Tips: specify GPUs/CPUs/memory per node in the script `slurm_train.sh` based on your resource!
