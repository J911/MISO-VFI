<h1 align="center">MISO-VFI: A Multi-In-Single-Out Network </br>for Video Frame Interpolation without Optical Flow</h1>
<h3 align="center" style="line-height:30px"><a href="https://cv.j911.me">Jaemin Lee</a>, <a href="https://sites.google.com/view/minseokcv">Minseok Seo</a>, <a  href="https://airlab.hanbat.ac.kr/theme/AiRLab_theme/member_cv/CV_sangwoo.pdf">Sangwoo Lee</a>, 
<br><a href="https://github.com/hbp001">Hyobin Park</a>, <a href="https://airlab.hanbat.ac.kr/members.php">Dong-Geol Choi</a></h3>

<h4 align="center"><a href="https://j911.github.io/MISO-VFI">Project Page</a>, <a href="https://arxiv.org/">Arxiv</a></h4>

### Abstract
<img align="center" style="width: 100%" src="./assets/teaser.jpg"></img>
<p align="justify">In general, deep learning-based video frame interpolation (VFI) methods have predominantly focused on estimating motion vectors between two input frames and warping them to the target time. While this approach has shown impressive performance for linear motion between two input frames, it exhibits limitations when dealing with occlusions and nonlinear movements. Recently, generative models have been applied to VFI to address these issues. However, as VFI is not a task focused on generating plausible images, but rather on predicting accurate intermediate frames between two given frames, performance limitations still persist. In this paper, we propose a multi-in-single-out (MISO) based VFI method that does not rely on motion vector estimation, allowing it to effectively model occlusions and nonlinear motion. Additionally, we introduce a novel motion perceptual loss that enables MISO-VFI to better capture the spatio-temporal correlations within the video frames. Our MISO-VFI method achieves state-of-the-art results on VFI benchmarks Vimeo90K, Middlebury, and UCF101, with a significant performance gap compared to existing approaches.</p>

### Citation
```bib
TBD
```

### How to use?
> **Code will be uploaded soon!**

#### Requirements
- pytorch == 1.10.0
- torchvision
- timm
- imageio-ffmpeg
- numpy

#### Datasets
**Vimeo90K datasets**
We used [Xue, Tianfan, et al.](http://toflow.csail.mit.edu/index.html)'s Dataset for the vimeo experiment.
```bash
$ mkdir /data
$ cd /data
$ wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip # (82G) # For 2-3-2
$ unzip vimeo_septuplet.zip
# or
$ wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip# (33G) # For 1-1-1
$ unzip vimeo_septuplet.zip
```

#### Training & Evaluation

**train**
```bash
$ python main.py --dataname vimeo --epochs 100 --batch_size 8 --test_batch_size 8 --vgg_loss True --perceptual True --mars_weight ./weights/kinetics-pretrained.pth --num_workers 16 --ex_name vimeo_exp --lr 0.001 --gpu 0,1,2,3 --data_root /data/vimeo_septuplet
```

`MARS Kinetics pre-trained weight` can get from [Google Drive](#) (we referred to the model of [MARS](https://github.com/craston/MARS).)

**evaluation**
```bash
$ python test.py --dataname vimeo --test_batch_size 4 --num_workers 16 --weight ./weights/vimeo-pretrained.pth --gpu 0,1,2,3 --data_root /data/vimeo_septuplet

# ---
# Results
# test mse:416.0105, mae:4615.9648, ssim:0.9552, psnr:37.19
```
If you want to get the vimeo90K pre-trained weight, Access this [Google Drive](#).

### Thanks to
- This code is heavily borrowed from [SimVP](https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction).
- We cite [MARS](https://github.com/craston/MARS)'s ResNeXt model.
- We cite metrics for SSIM and PSNR on [E3D-LSTM](https://github.com/google/e3d_lstm/blob/master/src/trainer.py).
- We use LKA-module of [Visual-Attention-Network](https://github.com/Visual-Attention-Network).
- We refinement learnable filter module of [MagNet](https://github.com/VinAIResearch/MagNet).
- We use ConvNeXt and LayerNorm of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
