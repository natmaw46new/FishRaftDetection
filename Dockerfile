# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
 
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git wget \
    libgdal-dev gdal-bin \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
 
RUN ln -s /usr/bin/python3.10 /usr/bin/python
 
WORKDIR /app
 
# torch first, in its own step - detectron2's setup.py imports torch at build time
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
 
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/detectron2.git
 
COPY requirements.txt .
RUN pip install --no-cache-dir --no-build-isolation -r requirements.txt
 
CMD ["tail", "-f", "/dev/null"]