# Project Objective

We propose to replicate and reconstruct the KJRD-Net using the PyTorch framework. Our implementation will focus on integrating the paper's key components, including a Vision Transformer (ViT) for feature extraction, FFA-Net for dehazing, RCAN for super-resolution, and an Image-Level Coordination Block (ICB) to consolidate these outputs. These components will work in tandem to enhance image quality, which will then be fed into a traditional Faster R-CNN for object detection. Initially, we will utilize the Faster R-CNN as the detection module, but we may consider substituting it with single-shot detectors like YOLO if the initial results demonstrate promise. The end-to-end model will be rigorously tested on at least two datasets, encompassing hazy images captured on land and underwater, to assess its robustness across diverse environments. 

## Pytorch installation.
Pytorch is required for this project.
Pls install pytroch in local environment following [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).