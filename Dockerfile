# RunPod Serverless용 Docker 이미지 (최적화)
# Wan 2.1 I2V 14B 720P 모델 - 경량 베이스 이미지 사용

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 및 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1 - 경량 wheel)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# AI/ML 패키지
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    imageio[ffmpeg] \
    Pillow \
    runpod

# 핸들러 복사
COPY runpod_handler.py /handler.py

CMD ["python", "/handler.py"]
