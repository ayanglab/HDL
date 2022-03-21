# HDL
This is the official implementation of our proposed HDL:

HDL: Hybrid Deep Learning for the Synthesis of Myocardial Velocity Maps in Digital Twins for Cardiac Analysis

![Overview_of_SwinMR](./tmp/FIG_Overview.png)


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
