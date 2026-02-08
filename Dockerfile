# RunPod Serverless용 Docker 이미지
# Wan 2.1 I2V 14B 720P 모델 (고화질 클라우드 전용)

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --no-cache-dir \
    diffusers>=0.36.0 \
    transformers>=4.44.0 \
    accelerate>=0.34.0 \
    sentencepiece \
    imageio[ffmpeg] \
    Pillow \
    runpod \
    protobuf

# 핸들러 복사
COPY runpod_handler.py /handler.py

# 모델을 미리 다운로드 (이미지 크기 커지지만 Cold Start 빠름)
# 주석 해제하면 빌드 시 모델 포함 (~30GB 이미지)
# RUN python -c "from diffusers import WanImageToVideoPipeline; WanImageToVideoPipeline.from_pretrained('Wan-AI/Wan2.1-I2V-14B-720P-Diffusers', torch_dtype='auto')"

# 핸들러 실행
CMD ["python", "/handler.py"]
