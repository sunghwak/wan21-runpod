# RunPod Serverless용 Docker 이미지 (초경량)
# python:3.11-slim 기반 - torch wheel에 CUDA 런타임 포함

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# ffmpeg만 설치 (비디오 인코딩용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1 런타임 포함된 wheel)
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
