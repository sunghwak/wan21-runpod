# CogVideoX-5B I2V - RunPod Serverless Docker Image
# Model: THUDM/CogVideoX-5b-I2V (~10GB)
# Base: python:3.11-slim (~150MB, pytorch 이미지 대비 ~4GB 절약)

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (ffmpeg for video encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1) - torchvision/torchaudio 제외하여 ~1.5GB 절약
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# ML dependencies
# - diffusers >= 0.32.0: CogVideoXImageToVideoPipeline 지원
# - transformers >= 4.44.0: CogVideoX 모델 지원
# - sentencepiece: T5 텍스트 인코더 토크나이저 필수
# - accelerate: 모델 로드 가속
RUN pip install --no-cache-dir \
    "diffusers>=0.32.0" \
    "transformers>=4.44.0" \
    accelerate \
    sentencepiece \
    ftfy \
    "imageio[ffmpeg]" \
    Pillow \
    runpod

COPY runpod_handler.py /app/

CMD ["python", "-u", "runpod_handler.py"]
