# CDVD-TSPNL

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/xuboming8/CDVD-TSPNL/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.0-%237732a8)](https://pytorch.org/)

#### [Paper](https://ieeexplore.ieee.org/document/10039490) | [Discussion](https://github.com/xuboming8/CDVD-TSPNL/issues)

### Cascaded Deep Video Deblurring Using Temporal Sharpness Prior and Non-local Spatial-Temporal Similarity
By [Jinshan Pan](https://jspan.github.io/), [Boming Xu](https://scholar.google.com.hk/citations?user=wdxQ8lwAAAAJ&hl=zh-CN), [Haoran Bai](https://scholar.google.com.hk/citations?user=SBYJ6YoAAAAJ&hl=zh-CN), [Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl=zh-CN), and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)

　　　　　![GOPRO](https://github.com/xuboming8/CDVD-TSPNL/assets/20449507/90bc043e-707a-49e5-828f-058d66a9d0ac)　
![DVD](https://github.com/xuboming8/CDVD-TSPNL/assets/20449507/c9280937-966e-468f-936b-27bc92e04d18)
<br />　　　　　　　　　　　　　**GOPRO**　　　　　　　　　　　　　　　　　　　**DVD**

　　　　　![BSD1](https://github.com/xuboming8/CDVD-TSPNL/assets/20449507/18b70ee4-febf-40f9-9264-c8f107b04891)　 
![BSD2](https://github.com/xuboming8/CDVD-TSPNL/assets/20449507/4f04bfff-7c3b-4371-8c1e-287e0405a60d)
<br />　　　　　　　　　　　　　　　　　　　　　　　**BSD(1ms8ms)**
<br /><br />This repository is the official PyTorch implementation of our TPAMI paper"Cascaded Deep Video Deblurring Using Temporal Sharpness Prior and Non-local Spatial-Temporal Similarity"


## Updates
[2022-04-30] Paper has been submitted to IEEE TPAMI\
[2023-01-23] Paper has been accepted by IEEE TPAMI\
[2023-01-31] Training & Testing code is available!

## Experimental Results
Deblurred result on a real challenging video. Our method is motivated by the success of deblurring
approaches based on variational models. It explores sharp pixels from adjacent frames by a temporal sharpness
prior and non-local spatial-temporal similarity contexts to constrain deep convolutional neural networks (CNNs)
and restores sharp videos by a cascaded inference process. We show that enforcing the temporal sharpness
prior and non-local spatial-temporal similarity contexts and learning the deep CNNs by a cascaded inference
manner can make the deep CNN more compact and generate better-deblurred results (i.e., (f) and (g)) than both
the CNN-based methods [2], [3], [4] and variational model-based method [1].
[![CDVD-TSPNL](https://s1.ax1x.com/2023/01/26/pSNRWHH.png)](https://imgse.com/i/pSNRWHH)


Quantitative evaluations on the DVD dataset in terms of PSNR and SSIM. * denotes the reported results using self-ensemble strategy.
[![DVD](https://s1.ax1x.com/2023/01/26/pSN2pon.png)](https://imgse.com/i/pSN2pon)

Quantitative evaluations on the GoPro dataset in terms of PSNR and SSIM. * denotes the reported results using self-ensemble strategy
[![GOPRO](https://s1.ax1x.com/2023/01/26/pSN2FzT.png)](https://imgse.com/i/pSN2FzT)

Quantitative evaluations on the BSD video deblurring dataset in terms of PSNR and SSIM.
[![BSD](https://s1.ax1x.com/2023/01/26/pSN2AQU.png)](https://imgse.com/i/pSN2AQU)

## Dependencies
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch 1.8.0](https://pytorch.org/): `conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch`
- Install dependent packages :`pip install -r requirements.txt`
- Install CDVD-TSPNL :`python setup.py develop`

## Get Started

### Pretrained models
- Models are available in  `'./experiments/pretrained_models/'`

### Dataset Organization Form
If you prepare your own dataset, please follow the following form like GOPRO/DVD:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```

### Training
- FlowNet pretrained model has been downloaded in `'./pretrained_models/flownet/'`
- Download training dataset like above form.
- Run the following commands:
```
Single GPU
python basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml
Multi-GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml --launcher pytorch
```

### Testing
- Models are available in  `'./experiments/pretrained_models/'`.
- Organize your dataset(GOPRO/DVD/BSD) like the above form.
- Run the following commands:
```
python basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO.yml
cd results
python merge_full.py
```
- Before running merge_full.py, you should change the parameters in this file of Line 5,6,7,8.
- The deblured result will be in `'./results/dataset_name/'`.
- If you set `flip_seq: Ture` in config files, testing code will use self-ensemble strategy.(CDVDTSPNL+)
- We calculate PSNRs/SSIMs following [[Here]](https://github.com/csbhr/OpenUtility#chapter-calculating-metrics)

## Citation
```
@InProceedings{Pan_2023_TPAMI,
    author = {Pan, Jinshan and Xu, Boming and Bai, Haoran and Tang, Jinhui and Yang, Ming-Hsuan},
    title = {Cascaded Deep Video Deblurring Using Temporal Sharpness Prior and Non-local Spatial-Temporal Similarity},
    booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence(TPAMI)},
    month = {Jan},
    year = {2023}
}
```

## Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.
