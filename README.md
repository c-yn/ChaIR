# Exploring the potential of channel interactions for image restoration
Yuning Cui, Alois Knoll

[![](https://img.shields.io/badge/Paper-blue.svg)](https://www.sciencedirect.com/science/article/abs/pii/S0950705123009061)
(Training details can be found in the supplementary material, which can also be downloaded from this link.)

>Image restoration aims to reconstruct a clear image from a degraded observation. Convolutional neural networks have achieved promising performance on this task. The usage of Transformer has recently made significant advancements in state-of-the-art performance by modeling long-range dependencies. However, these deep architectures primarily concentrate on enhancing representation learning for the spatial dimension, neglecting the significance of channel interactions. In this paper, we explore the potential of channel interactions for restoring images through our proposal of a dual-domain channel attention mechanism. To be specific, channel attention in the spatial domain allows each channel to amass valuable signals from adjacent channels under the guidance of learned dynamic weights. In order to effectively exploit the significant difference in infrequency between degraded and clean image pairs, we develop the implicit frequency domain channel attention to facilitate the integration of information from different frequencies. Extensive experiments demonstrate that the proposed network, dubbed ChaIR, achieves state-of-the-art performance on 13 benchmark datasets for five image restoration tasks, including image dehazing, image motion/defocus deblurring, image desnowing, and image deraining.


## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation
Please refer to respective directories.
## Results (ChaIR) 
### Images [here](https://drive.google.com/drive/folders/1PEyjMOJ8-MAZOkS1manAF-0cyWABiMJW?usp=drive_link)

|Task|Dataset|PSNR|SSIM|
|----|------|-----|----|
|**Motion Deblurring**|GoPro|33.28|0.963|
||HIDE|30.97|0.941|
||RSBlur|34.25|0.871|
|**Image Dehazing**|SOTS-Indoor|41.95|0.997|
||SOTS-Outdoor|40.73|0.997|
||Dense-Haze|17.50|0.62|
||NHR|28.18|0.98|
|**Image Desnowing**|CSD|39.24|0.99|
||SRRS|31.91|0.98|
||Snow100K|33.79|0.95|
|**Image Deraining**|Rain100L|38.20|0.973|
||Rain100H|31.74|0.906|
|**Defocus Deblurring**|DPDD|26.29|0.816|



## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{cui2023exploring,
  title={Exploring the potential of channel interactions for image restoration},
  author={Cui, Yuning and Knoll, Alois},
  journal={Knowledge-Based Systems},
  volume={282},
  pages={111156},
  year={2023},
  publisher={Elsevier}
}
~~~
## Contact
Should you have any question, please contact Yuning Cui.
