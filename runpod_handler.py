"""
RunPod Serverless Handler - Wan 2.1 Image-to-Video (720P)
RunPod Serverless 엔드포인트에 배포하는 핸들러 스크립트

⚡ 720P 고화질 버전 - RunPod 클라우드 GPU (48GB+) 전용
   로컬(RTX 3070)은 480P, 클라우드는 720P로 고품질 생성!

배포 방법:
1. RunPod 계정 생성: https://www.runpod.io
2. Serverless → New Endpoint 생성
3. GitHub 연동으로 자동 빌드/배포
4. GPU Type: A40/A6000 (48GB) 이상 권장
"""

import io
import os
import time
import base64
import logging
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# 전역 파이프라인 (cold start 시 한 번만 로드)
pipe = None
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

SUPPORTED_RESOLUTIONS = {
    "16:9": (720, 1280),
    "4:3": (720, 960),
    "1:1": (720, 720),
    "9:16": (1280, 720),
    "3:4": (960, 720),
}


def load_model():
    """모델을 GPU에 로드 (80GB A100 - 720P 모델)"""
    global pipe
    if pipe is not None:
        return

    from diffusers import WanImageToVideoPipeline

    # Network Volume에 모델 캐시 (/runpod-volume 마운트됨)
    cache_dir = "/runpod-volume/huggingface"
    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"모델 로딩 시작: {MODEL_ID} (캐시: {cache_dir})")
    start = time.time()

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    # 80GB GPU - 전체 모델을 GPU에 로드 (최대 성능)
    pipe.to("cuda")

    elapsed = time.time() - start
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    logger.info(f"모델 로드 완료! 시간: {elapsed:.1f}초, GPU: {gpu_name}")


def prepare_image(image_b64: str, resolution: str = "auto") -> tuple:
    """base64 이미지를 PIL Image로 변환 및 리사이즈"""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if resolution == "auto":
        w, h = image.size
        aspect = w / h
        best = min(
            SUPPORTED_RESOLUTIONS.items(),
            key=lambda x: abs((x[1][1] / x[1][0]) - aspect),
        )
        target_h, target_w = best[1]
        resolution = best[0]
    elif resolution in SUPPORTED_RESOLUTIONS:
        target_h, target_w = SUPPORTED_RESOLUTIONS[resolution]
    else:
        target_h, target_w = 480, 832
        resolution = "16:9"

    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image, target_h, target_w, resolution


def handler(job):
    """
    RunPod Serverless 핸들러

    Input:
        image_base64: str - base64 인코딩된 입력 이미지
        prompt: str - 동작/장면 설명 프롬프트
        negative_prompt: str - 네거티브 프롬프트 (선택)
        num_frames: int - 생성할 프레임 수 (기본: 33)
        num_inference_steps: int - 추론 스텝 (기본: 25)
        guidance_scale: float - 가이던스 스케일 (기본: 5.0)
        seed: int - 시드 (기본: -1, 랜덤)
        resolution: str - 해상도 (기본: auto)
        fps: int - FPS (기본: 16)

    Output:
        video_base64: str - base64 인코딩된 MP4 비디오
        generation_time: float - 생성 시간(초)
        resolution: str - 출력 해상도
        gpu_name: str - 사용된 GPU
        num_frames: int - 실제 생성 프레임 수
    """
    # 모델 로드 (cold start 시만)
    load_model()

    input_data = job["input"]

    # 파라미터 추출
    image_b64 = input_data.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 is required"}

    prompt = input_data.get("prompt", "")
    negative_prompt = input_data.get("negative_prompt", "")
    num_frames = int(input_data.get("num_frames", 33))
    num_inference_steps = int(input_data.get("num_inference_steps", 25))
    guidance_scale = float(input_data.get("guidance_scale", 5.0))
    seed = int(input_data.get("seed", -1))
    resolution = input_data.get("resolution", "auto")
    fps = int(input_data.get("fps", 16))

    try:
        # 이미지 전처리
        image, height, width, res_name = prepare_image(image_b64, resolution)

        # 시드 설정
        if seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # VRAM 초기화
        torch.cuda.reset_peak_memory_stats()

        # 생성
        logger.info(
            f"생성 시작: {width}x{height}, {num_frames}프레임, "
            f"{num_inference_steps}스텝, seed={seed}"
        )
        start_time = time.time()

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        generation_time = time.time() - start_time

        # 비디오를 base64로 인코딩
        from diffusers.utils import export_to_video

        tmp_path = "/tmp/output_video.mp4"
        export_to_video(output.frames[0], tmp_path, fps=fps)

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # 정리
        os.remove(tmp_path)
        vram_peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        gpu_name = torch.cuda.get_device_name(0)

        logger.info(
            f"생성 완료! 시간: {generation_time:.1f}초, "
            f"VRAM: {vram_peak}GB, GPU: {gpu_name}"
        )

        return {
            "video_base64": video_b64,
            "generation_time": round(generation_time, 1),
            "resolution": f"{width}x{height} ({res_name})",
            "gpu_name": gpu_name,
            "vram_peak_gb": vram_peak,
            "num_frames": num_frames,
            "seed": seed,
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "error": "GPU 메모리 부족. 프레임 수나 해상도를 줄여주세요.",
            "generation_time": 0,
        }
    except Exception as e:
        logger.error(f"생성 실패: {e}", exc_info=True)
        return {"error": str(e), "generation_time": 0}


# RunPod Serverless 시작
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
