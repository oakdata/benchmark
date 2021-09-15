# Wanderlust: Online Continual Object Detection in the Real World

By: Jianren Wang, Xin Wang, Yue Shang-Guan, Abhinav Gupta

This repository is an official implementation of the paper [Wanderlust: Online Continual Object Detection in the Real World](https://arxiv.org/abs/2108.11005)

## Introduction

**TL; DR.** Objects Around Krishna (OAK) is a new egocentric video dataset for online continual object detection. We provide a benchmark where the emergence of new object categories follows a pattern similar to what a single person will see in their everyday life.

**Abstract.** Online continual learning from data streams in dynamic environments is a critical direction in the computer vision field. However, realistic benchmarks and fundamental studies in this line are still missing. To bridge the gap, we present a new online continual object detection benchmark with an egocentric video dataset, Objects Around Krishna (OAK). The emergence of new object categories in our benchmark follows a pattern similar to what a single person might see in their day-to-day life. The dataset also captures the natural distribution shifts as the person travels to different places. These egocentric long running videos provide a realistic playground for continual learning algorithms, especially in online embodied settings. 

## Repository Structure
    root
    ├── object_detection                        # Object Detection Benchmark
        ├── detectron2                          # Benchmark using Faster RCNN Architecture
        └── deformable_detr (coming soon)
    ├── image_classification (coming soon)      # Image Classification Benchmark
    ├── LICENSE
    └── README.md

We provide both the `object_detection` and `image_classification` benchmarks using OAK.

## Reproducing Results
