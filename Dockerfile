# Wan2.2-I2V-A14B - RunPod Serverless Docker Image
# Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers (~126GB, pre-downloaded on Network Volume)
# MoE I2V with Lightning LoRA acceleration
# Base: nvidia/cuda:12.8.0-runtime-ubuntu22.04

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Model is loaded from Network Volume, not HF cache
ENV HF_HOME=/runpod-volume/huggingface

WORKDIR /app

# System dependencies (ffmpeg for video encoding, git for pip installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.8)
RUN pip3 install --no-cache-dir \
    torch==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Diffusers fork with Wan2.2 MoE LoRA support
# This fork has WanImageToVideoPipeline with transformer_2 + load_into_transformer_2
RUN pip3 install --no-cache-dir \
    "git+https://github.com/linoytsaban/diffusers.git@wan22-loras"

# ML dependencies
RUN pip3 install --no-cache-dir \
    "transformers>=4.46.0" \
    "accelerate>=1.0.0" \
    "peft>=0.15.0" \
    "safetensors>=0.4.0" \
    sentencepiece \
    ftfy \
    "imageio[ffmpeg]" \
    Pillow \
    numpy

# Quantization support (text encoder Int8)
RUN pip3 install --no-cache-dir \
    torchao

# RunPod SDK
RUN pip3 install --no-cache-dir \
    runpod

# Copy handler
COPY runpod_handler.py /app/

CMD ["python3", "-u", "runpod_handler.py"]
