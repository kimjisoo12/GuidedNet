# GuidedNet: Semi-Supervised Multi-Organ Segmentation via Labeled Data Guide Unlabeled Data

Official PyTorch implementation of [GuidedNet: Semi-Supervised Multi-Organ Segmentation via Labeled Data Guide Unlabeled Data](https://arxiv.org/abs/2408.04914), ACM MM 2024.

> **Abstract.** 
Semi-supervised multi-organ medical image segmentation aids physicians in improving disease diagnosis and treatment planning and reduces the time and effort required for organ annotation. 
Existing state-of-the-art methods train the labeled data with ground truths and train the unlabeled data with pseudo-labels. However, the two training flows are separate, which does not reflect the interrelationship between labeled and unlabeled data.	
To address this issue, we propose a semi-supervised multi-organ segmentation method called GuidedNet, which leverages the knowledge from labeled data to guide the training of unlabeled data. The primary goals of this study are to improve the quality of pseudo-labels for unlabeled data and to enhance the network's learning capability for both small and complex organs.
A key concept is that voxel features from labeled and unlabeled data that are close to each other in the feature space are more likely to belong to the same class. 
On this basis, a 3D Consistent Gaussian Mixture Model (3D-CGMM) is designed to leverage the feature distributions from labeled data to rectify the generated pseudo-labels.
Furthermore, we introduce a Knowledge Transfer Cross Pseudo Supervision (KT-CPS) strategy, which leverages the prior knowledge obtained from the labeled data to guide the training of the unlabeled data, thereby improving the segmentation accuracy for both small and complex organs.
Extensive experiments on two public datasets, FLARE22 and AMOS, demonstrated that GuidedNet is capable of achieving state-of-the-art performance.

![](https://github.com/kimjisoo12/GuidedNet/blob/main/thumbnail%20image.jpg)

## Usage
### Requirements
This repository is based on PyTorch 1.7.1, CUDA 10.1 and Python 3.9.7; All experiments in our paper were conducted on on two NVIDIA A100 GPUs (40G).

### Dataset 
The datasets used in our paper are [FLARE22 dataset](https://arxiv.org/abs/2308.05862) and [AMOS dataset](https://arxiv.org/abs/2206.08023). 
Preprocessed data can be found [here](https://pan.baidu.com/s/1aUDUu3iVfrlrVidcJmOkcg?pwd=1fvb) , Fetch Code: 1fvb.

### Training Steps
1. Clone the repo and create data path:
```
git clone https://github.com/kimjisoo12/GuidedNet.git
cd GuidedNet
mkdir data # create data path
```
2. Put the preprocessed data and split.txt in ./data/flare for FLARE22 dataset. (./data/amos for AMOS dataset) and then cd code.
3. We train our model on two NVIDIA A100 GPUs (40G) for each dataset.

To produce the claimed results for FLARE22 dataset:
```
# For 10% labeled data,
CUDA_VISIBLE_DEVICES=0,1 python train_guidedNet_flare10.py --model 'guidedNet' --max_iterations 20000 --consistency 0.1 --base_lr 0.1 --batch_size 8 --labeled_bs 4 --lanmuda 0.3 --data_num 420 --labeled_num 42

# For 50% labeled data, 
CUDA_VISIBLE_DEVICES=0,1 python train_guidedNet_flare10.py --model 'guidedNet' --max_iterations 20000 --consistency 0.1 --base_lr 0.1 --batch_size 8 --labeled_bs 4 --lanmuda 0.3 --data_num 84 --labeled_num 42

```

To produce the claimed results for AMOS dataset:
```
# For 10% labeled data,
CUDA_VISIBLE_DEVICES=0,1 python train_guidedNet_amos.py --model 'guidedNet' --max_iterations 20000 --consistency 0.1 --base_lr 0.1 --batch_size 8 --labeled_bs 4 --lanmuda 0.3 --data_num 180 --labeled_num 18

# For 50% labeled data, 
CUDA_VISIBLE_DEVICES=0,1 python train_guidedNet_amos.py --model 'guidedNet' --max_iterations 20000 --consistency 0.1 --base_lr 0.1 --batch_size 8 --labeled_bs 4 --lanmuda 0.3 --data_num 180 --labeled_num 90

```
###  Checkpoints

Test data, label, checkpoints can be found [here](https://pan.baidu.com/s/1sLXk7eb6NuYdtSJw0FBOfg?pwd=4t9i) , Fetch Code: 4t9i 

###  Infer

1. Put the test data in ./infer/data/flare for FLARE22 dataset. (./infer/data/flare for AMOS dataset) 

2. Put the test laebl in ./infer/label/flare for FLARE22 dataset. (./infer/label/flare for AMOS dataset) 

3. Put the checkpoints in ./infer/checkpoints/flare/10/our or ./infer/checkpoints/flare/50/our for FLARE22 dataset. (./infer/checkpoints/amos/10/our or ./infer/checkpoints/amos/50/our for AMOS dataset) and then cd code.

Finally, the structure of dictionary ```infer``` should be as follows:

```angular2html
infer
├── data
│   ├── flare
│   └── amos
├── label
│   ├── flare
│   └── amos
└── checkpoints
    ├── flare
    │   ├── 10
    │   │   └── our
        ├── 50
    │   │   └── our
    └── amos
    │   ├── 10
    │   │   └── our
    │   └── 50
    │   │   └── our
```

To produce the infer test results for FLARE22 dataset:
```
# For 10% labeled data,
python predict_organ_flare.py --ratio 10

# For 50% labeled data, 
python predict_organ_flare.py --ratio 50

```

To produce the infer test results for AMOS dataset:
```
# For 10% labeled data,
python predict_organ_amos.py --ratio 10

# For 50% labeled data, 
python predict_organ_amos.py --ratio 50

```


## Citation
```bibtex
@inproceedings{10.1145/3664647.3681526,
author = {Zhao, Haochen and Meng, Hui and Yang, Deqian and Xie, Xiaozheng and Wu, Xiaoze and Li, Qingfeng and Niu, Jianwei},
title = {GuidedNet: Semi-Supervised Multi-Organ Segmentation via Labeled Data Guide Unlabeled Data},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681526},
doi = {10.1145/3664647.3681526},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {886–895},
numpages = {10},
keywords = {3d medical image segmentation, abdominal organs, gaussian mixture model, semi-supervised learning},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

## Contact

- studyzhao@buaa.edu.cn

## Poster
![](https://github.com/kimjisoo12/GuidedNet/blob/main/poster.png)
