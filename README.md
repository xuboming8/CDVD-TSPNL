# CDVD-TSPNL

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/xuboming8/CDVD-TSPNL/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.8.0-%237732a8)](https://pytorch.org/)

### Cascaded Deep Video Deblurring Using Temporal Sharpness Prior and Non-local Spatial-Temporal Similarity
By [Jinshan Pan](https://jspan.github.io/), Boming Xu, and [Haoran Bai](https://csbhr.github.io/)

This repository is the official PyTorch implementation of "Cascaded Deep Video Deblurring Using Temporal Sharpness Prior and Non-local Spatial-Temporal Similarity"

## Updates
[2022-02-08] Training code and Testing code are available!  
[2022-02-07] Paper coming soon...

## Experimental Results
Quantitative evaluations on the video deblurring dataset [11] in terms of PSNR and SSIM. All the comparison
results are generated using the publicly available code. All the restored frames instead of randomly selected 30
frames from each test set [11] are used for evaluations.
[![DVD](https://s4.ax1x.com/2022/02/07/HlpJU0.png)](https://imgtu.com/i/HlpJU0)

Quantitative evaluations on the GoPro dataset [43] in terms of PSNR and SSIM. * denotes the reported results
from [47].
[![GOPRO](https://s4.ax1x.com/2022/02/07/HlpGEq.png)](https://imgtu.com/i/HlpGEq)

Quantitative evaluations on the BSD video deblurring dataset [5] in terms of PSNR and SSIM.
[![BSD](https://s4.ax1x.com/2022/02/07/Hl9hWT.png)](https://imgtu.com/i/Hl9hWT)

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
- Model are available in  `'./experiments/pretrained_models/'`
- Organize your dataset(GOPRO/DVD/BSD) like the above form.
- Run the following commands:
```
python basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO.yml
```
- The deblured result will be in './results/'.
- We calculate PSNRs/SSIMs following [[Here]](https://github.com/csbhr/OpenUtility#chapter-calculating-metrics)
- If we set `flip_seq: Ture` in config files, testing code will use self-ensemble strategy.(CDVDTSPNL+)

## Citation