FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel


RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1 libgtk2.0-dev libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libfontconfig1

RUN pip install --upgrade pip setuptools wheel
RUN pip install timm==0.9.7 imageio imageio-ffmpeg numpy einops==0.6.1 opencv-python scikit-image

WORKDIR /ws