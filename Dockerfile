# Wan2.1-Fun-V1.3-InP - RunPod Serverless Docker Image
# Model: alibaba-pai/Wan2.1-Fun-V1.3-InP (~37GB)
# VideoX-Fun InPainting-based I2V (Wan 2.1 14B, uncensored)
# Base: python:3.11-slim

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (ffmpeg for video encoding, git for pip installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1) - torchvision/torchaudio excluded to save ~1.5GB
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu121

# ML dependencies
# - diffusers >= 0.34.0: WanFunInpaintPipeline support
# - transformers >= 4.46.0: Wan 2.1 model support
# - sentencepiece: T5 text encoder tokenizer
# - accelerate: model loading acceleration
RUN pip install --no-cache-dir \
    "diffusers>=0.34.0" \
    "transformers>=4.46.0" \
    accelerate \
    sentencepiece \
    ftfy \
    "imageio[ffmpeg]" \
    Pillow \
    numpy \
    runpod

# Install VideoX-Fun package (provides WanFunInpaintPipeline if not in diffusers)
RUN pip install --no-cache-dir "git+https://github.com/aigc-apps/VideoX-Fun.git" || \
    echo "VideoX-Fun install failed (optional, diffusers may have built-in support)"

COPY runpod_handler.py /app/

CMD ["python", "-u", "runpod_handler.py"]
