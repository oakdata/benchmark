# Deformable DETR Baseline

The architecture is based on the official implementation of the paper [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159).
By [Xizhou Zhu](https://scholar.google.com/citations?user=02RXI00AAAAJ),  [Weijie Su](https://www.weijiesu.com/),  [Lewei Lu](https://www.linkedin.com/in/lewei-lu-94015977/), [Bin Li](http://staff.ustc.edu.cn/~binli/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Jifeng Dai](https://jifengdai.org/).

## Introduction

Deformable DETR is an efficient and fast-converging end-to-end object detector which mitigates the high complexity and slow convergence issues of DETR via a novel sampling-based efficient attention mechanism. We leverage this architecture and evaluate its performance in a continual object detection setting in five main schemes: iCaRL, EWC, Incremental, Offline, Non-Adaptation.

## Installation

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

Please download [OAK](https://oakdata.github.io) and organize them as following:

```
code_root/
└── data/
    └── oak/
        ├── train/
        └── test/
```

### Training

#### Training on single node

For example, the command for training Deformable DETR on 8 GPUs is as following:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh --output_dir=output/incremental_oak_1 --train_mode=incremental --batch_size=2 --iterations=10 --dataset_file=oak --lr=1e-4 --lr_backbone=1e-5 --lr_drop=1000
```

#### Training on slurm cluster

If you are using slurm cluster, you can simply run the following command to train on 1 node with 8 GPUs, while resuming from specific checkpoint (if resume is not needed, simply remove the flag):

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm_grogu.sh abhinav deform_detr 8 ./configs/r50_deformable_detr.sh --output_dir=output/incremental_oak_1 --train_mode=incremental --batch_size=2 --iterations=10 --dataset_file=oak --oak_path=/grogu/user/jianrenw/data --lr=1e-4 --lr_backbone=1e-5 --lr_drop=1000 --resume /path/to/checkpoint/checkpoint.pth
```

#### Some tips to speed-up training
* You may increase the batch size to maximize the GPU utilization, according to GPU memory of yours, e.g., set '--batch_size 3' or '--batch_size 4'.

### Evaluation


## Citing Wanderlust
If you find this baseline useful in your research, please consider citing
```bibtex
@misc{wang2021wanderlust,
      title={Wanderlust: Online Continual Object Detection in the Real World}, 
      author={Jianren Wang and Xin Wang and Yue Shang-Guan and Abhinav Gupta},
      year={2021},
      eprint={2108.11005},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Citing Deformable DETR
If you find Deformable DETR useful in your research, please consider citing:
```bibtex
@article{zhu2020deformable,
  title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
  author={Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  journal={arXiv preprint arXiv:2010.04159},
  year={2020}
}
```
