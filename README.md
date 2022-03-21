# HDL
This is the official implementation of our proposed HDL:

HDL: Hybrid Deep Learning for the Synthesis of Myocardial Velocity Maps in Digital Twins for Cardiac Analysis

Synthetic digital twins based on medical data accelerate the acquisition, labelling and decision making procedure in digital healthcare. A core part of digital healthcare twins is model-based data synthesis, which permits  the generation of realistic medical signals without requiring to cope with the modelling complexity of anatomical and biochemical phenomena producing them in reality. Unfortunately, algorithms for cardiac data synthesis have been so far scarcely studied. An important imaging modality in the cardiac examination is three-directional CINE multi-slice myocardial velocity mapping (3Dir MVM), which provides a quantitative assessment of cardiac motion in three orthogonal directions of the left ventricle. The long acquisition time and complex acquisition produce make it more urgent to produce synthetic digital twins of this imaging modality. In this study, we propose a hybrid deep learning (HDL) network, especially for synthetic 3Dir MVM data. Our algorithm is featured by a hybrid UNet and a Generative Adversarial Network with a foreground-background generation scheme.

![Overview_of_HDL](./img/Fig2.png)
![GraphicAbstract](./img/Fig1.png)


## Highlight

- A UNet model for cardiac frame interpolation.
- A foreground-background generation scheme for cardiac phase images.



## Requirements

matplotlib==3.3.4

opencv-python==4.5.3.56

Pillow==8.3.2

pytorch-fid==0.2.0

scikit-image==0.17.2

scipy==1.5.4

torch==1.9.0

torchvision==0.10.0


## Citation
This repository is based on:

pix2pixHD: Image Restoration Using Swin Transformer ([code](https://github.com/NVIDIA/pix2pixHD) and 
[paper](https://arxiv.org/abs/1711.11585));



Paper Link:

https://arxiv.org/abs/2203.05564 \
https://ieeexplore.ieee.org/document/9735339

Please cite:

```
@ARTICLE{9735339,
  author={Xing, Xiaodan and Del Ser, Javier and Wu, Yinzhe and Li, Yang and Xia, Jun and Lei, Xu and Firmin, David and Gatehouse, Peter and Yang, Guang},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={HDL: Hybrid Deep Learning for the Synthesis of Myocardial Velocity Maps in Digital Twins for Cardiac Analysis}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JBHI.2022.3158897}}
```
