"""
RunPod Serverless Handler - CogVideoX1.5-5B I2V
Image-to-Video generation using CogVideoX1.5-5B model (THUDM/Tsinghua)

Model: THUDM/CogVideoX1.5-5b-I2V
Resolution: 768x1360 (default), custom supported
Frames: 81 (5s@16fps) or 161 (10s@16fps)
VRAM: ~24GB (A100 80GB recommended)
"""

import os
import io
import json
import time
import base64
import tempfile
import traceback
import shutil

import torch
from PIL import Image

import runpod

# --- Configuration ---
MODEL_ID = "THUDM/CogVideoX1.5-5b-I2V"
CACHE_DIR = "/runpod-volume/huggingface"

# 캐시 버전: 이 값을 변경하면 Network Volume의 모델 캐시가 강제 삭제 후 재다운로드됨
# v1: CogVideoX 1.5 최초 배포
CACHE_VERSION = "cogvideox1.5-5b-i2v-v1"

# CogVideoX1.5-5B safetensors 최소 크기 기준
# 실제 파일은 수백MB~수GB이므로 10MB 미만이면 불완전 다운로드로 판단
SAFETENSORS_MIN_SIZE_MB = 10

# CogVideoX1.5-5B-I2V 기본 해상도 (공식 권장: 768x1360)
DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 1360

# CogVideoX1.5 지원 해상도 (height, width) - I2V는 커스텀 해상도도 지원
SUPPORTED_RESOLUTIONS = {
    "16:9": (768, 1360),   # 기본 권장
    "9:16": (1360, 768),   # 세로형
    "4:3": (768, 1024),    # 4:3
    "3:4": (1024, 768),    # 세로 4:3
    "1:1": (768, 768),     # 정사각형
}

pipe = None
gpu_name = "Unknown"


def get_cache_marker_path(cache_dir):
    """캐시 버전 마커 파일 경로 반환"""
    return os.path.join(cache_dir, ".cache_version")


def needs_force_clean(cache_dir):
    """버전 마커를 확인하여 강제 재다운로드가 필요한지 판단"""
    marker_path = get_cache_marker_path(cache_dir)

    if not os.path.exists(marker_path):
        print(f"[Cache] No version marker found - force clean needed")
        return True

    with open(marker_path, "r") as f:
        stored_version = f.read().strip()

    if stored_version != CACHE_VERSION:
        print(f"[Cache] Version mismatch: stored='{stored_version}' vs expected='{CACHE_VERSION}'")
        return True

    print(f"[Cache] Version marker matches: {CACHE_VERSION}")
    return False


def mark_cache_valid(cache_dir):
    """모델 로드 성공 후 버전 마커 기록"""
    marker_path = get_cache_marker_path(cache_dir)
    with open(marker_path, "w") as f:
        f.write(CACHE_VERSION)
    print(f"[Cache] Version marker written: {CACHE_VERSION}")


def force_clean_model_cache(cache_dir, model_id):
    """모델 캐시를 완전히 삭제하여 깨끗한 재다운로드 보장"""
    model_cache_path = os.path.join(
        cache_dir, "models--" + model_id.replace("/", "--")
    )

    if os.path.exists(model_cache_path):
        # 기존 캐시 크기 계산
        total_size = 0
        file_count = 0
        for dp, dn, filenames in os.walk(model_cache_path):
            for f in filenames:
                total_size += os.path.getsize(os.path.join(dp, f))
                file_count += 1

        print(f"[Cache] FORCE CLEAN: Removing {file_count} files ({total_size / (1024**3):.2f} GB)")
        print(f"[Cache] Path: {model_cache_path}")
        shutil.rmtree(model_cache_path, ignore_errors=True)
        print(f"[Cache] Removed successfully. Will re-download from HuggingFace.")
    else:
        print(f"[Cache] No existing cache to clean at: {model_cache_path}")

    # 버전 마커도 삭제
    marker_path = get_cache_marker_path(cache_dir)
    if os.path.exists(marker_path):
        os.remove(marker_path)


def validate_cache(cache_dir, model_id):
    """Network Volume에 캐시된 모델 파일의 무결성을 철저히 검증"""
    if not os.path.exists(cache_dir):
        print(f"[Cache] Cache directory does not exist: {cache_dir}")
        return False

    model_cache_path = os.path.join(
        cache_dir, "models--" + model_id.replace("/", "--")
    )
    if not os.path.exists(model_cache_path):
        print(f"[Cache] No cached model found at: {model_cache_path}")
        return False

    print(f"[Cache] Validating cache at: {model_cache_path}")
    corrupted = False
    safetensors_files = []
    total_safetensors_size = 0

    for root, dirs, files in os.walk(model_cache_path):
        for f in files:
            filepath = os.path.join(root, f)

            # JSON 파일 검증
            if f.endswith(".json"):
                try:
                    with open(filepath, "r") as fh:
                        json.load(fh)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    print(f"[Cache] CORRUPTED JSON: {filepath}")
                    corrupted = True
                    break

            # safetensors 파일 검증 (무결성 핵심)
            elif f.endswith(".safetensors"):
                file_size = os.path.getsize(filepath)
                file_size_mb = file_size / (1024 * 1024)
                safetensors_files.append((f, file_size_mb))
                total_safetensors_size += file_size

                if file_size_mb < SAFETENSORS_MIN_SIZE_MB:
                    print(f"[Cache] CORRUPTED safetensors (too small: {file_size_mb:.1f}MB): {filepath}")
                    corrupted = True
                    break

            # 기타 바이너리 모델 파일
            elif f.endswith((".model", ".bin")):
                file_size = os.path.getsize(filepath)
                if file_size < 1024:
                    print(f"[Cache] CORRUPTED binary (too small: {file_size}B): {filepath}")
                    corrupted = True
                    break

        if corrupted:
            break

    # safetensors 총 크기 검증 (CogVideoX1.5-5B = 약 10GB 이상이어야 함)
    if not corrupted:
        total_gb = total_safetensors_size / (1024**3)
        print(f"[Cache] Found {len(safetensors_files)} safetensors files, total: {total_gb:.2f} GB")

        for name, size_mb in safetensors_files:
            print(f"  - {name}: {size_mb:.1f} MB")

        if total_gb < 5.0:
            print(f"[Cache] CORRUPTED: Total safetensors size {total_gb:.2f}GB < 5GB minimum!")
            corrupted = True

    if corrupted:
        print(f"[Cache] Corrupted cache detected! Removing for clean re-download...")
        shutil.rmtree(model_cache_path, ignore_errors=True)
        # 마커도 삭제
        marker_path = get_cache_marker_path(cache_dir)
        if os.path.exists(marker_path):
            os.remove(marker_path)
        print(f"[Cache] Removed: {model_cache_path}")
        return False
    else:
        print(f"[Cache] All validations passed!")
        return True


def cleanup_old_models(cache_dir, current_model_id):
    """이전 모델 캐시를 삭제하여 디스크 공간 확보"""
    if not os.path.exists(cache_dir):
        return

    current_cache_name = "models--" + current_model_id.replace("/", "--")

    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if item.startswith("models--") and item != current_cache_name and os.path.isdir(item_path):
            print(f"[Cleanup] Removing old model cache: {item}")
            shutil.rmtree(item_path, ignore_errors=True)
            print(f"[Cleanup] Removed: {item}")

    # Also clean up any orphaned temp/lock files
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if item.endswith(".lock") or item.endswith(".tmp"):
            try:
                os.remove(item_path)
                print(f"[Cleanup] Removed temp file: {item}")
            except OSError:
                pass


def load_model():
    """CogVideoX1.5-5B I2V 모델 로드"""
    global pipe, gpu_name

    if pipe is not None:
        return

    print(f"[Model] Loading {MODEL_ID}...")
    print(f"[Model] Cache version: {CACHE_VERSION}")
    start = time.time()

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Model] GPU: {gpu_name} ({vram_gb:.1f} GB)")

    # 캐시 디렉토리 생성
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: 이전 모델(CogVideoX 1.0, Wan 2.1 등) 캐시 삭제
    cleanup_old_models(CACHE_DIR, MODEL_ID)

    # Step 2: 버전 마커 확인 → 불일치 시 강제 전체 삭제
    if needs_force_clean(CACHE_DIR):
        print(f"[Cache] === FORCE CLEAN MODE ===")
        force_clean_model_cache(CACHE_DIR, MODEL_ID)
    else:
        # 마커 일치 시에도 safetensors 무결성 검증
        validate_cache(CACHE_DIR, MODEL_ID)

    # Step 3: 디스크 공간 확인
    disk_stat = shutil.disk_usage(CACHE_DIR)
    free_gb = disk_stat.free / (1024**3)
    total_gb = disk_stat.total / (1024**3)
    print(f"[Disk] Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB")
    if free_gb < 2.0:
        print(f"[Disk] WARNING: Very low disk space! Model download may fail.")

    # Step 4: 모델 로드 (없으면 자동 다운로드)
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler

    print(f"[Model] Loading from HuggingFace (cache: {CACHE_DIR})...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE_DIR,
    )

    # 공식 권장 스케줄러: CogVideoXDPMScheduler + timestep_spacing="trailing"
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    print(f"[Model] Scheduler: CogVideoXDPMScheduler (timestep_spacing=trailing)")

    # CogVideoX 3D VAE 필수 설정
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print(f"[Model] VAE tiling + slicing enabled")

    # GPU 배치: A100-80GB면 full GPU, 아니면 CPU offload
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb >= 40:
            pipe.to("cuda")
            print(f"[Model] Full GPU mode ({vram_gb:.0f}GB VRAM)")
        else:
            pipe.enable_model_cpu_offload()
            print(f"[Model] CPU offload mode ({vram_gb:.0f}GB VRAM)")

    elapsed = time.time() - start
    print(f"[Model] Loaded in {elapsed:.1f}s")

    # Step 5: 로드 성공 → 버전 마커 기록
    mark_cache_valid(CACHE_DIR)
    print(f"[Model] Ready for inference!")


def prepare_image(image_b64, target_height=None, target_width=None):
    """base64 이미지를 PIL Image로 변환하고 지정 해상도로 리사이즈

    CogVideoX1.5-5B-I2V는 커스텀 해상도 지원.
    기본값: 768x1360 (height x width)
    """
    # Decode base64
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    orig_w, orig_h = image.size
    print(f"[Image] Original size: {orig_w}x{orig_h}")

    # 목표 해상도 결정
    if target_height is None:
        target_height = DEFAULT_HEIGHT
    if target_width is None:
        target_width = DEFAULT_WIDTH

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"[Image] Resized to: {target_width}x{target_height}")

    return image


def resolve_resolution(resolution_str):
    """해상도 문자열을 (height, width) 튜플로 변환

    지원 형식:
    - "auto" 또는 "16:9" → 기본 권장 해상도 (768x1360)
    - "9:16", "4:3", "3:4", "1:1" → SUPPORTED_RESOLUTIONS에서 조회
    - "HxW" 형식 (예: "768x1360") → 직접 파싱
    """
    if not resolution_str or resolution_str == "auto":
        return DEFAULT_HEIGHT, DEFAULT_WIDTH

    # SUPPORTED_RESOLUTIONS에서 조회
    if resolution_str in SUPPORTED_RESOLUTIONS:
        return SUPPORTED_RESOLUTIONS[resolution_str]

    # "HxW" 형식 파싱
    if "x" in resolution_str:
        try:
            parts = resolution_str.lower().split("x")
            h, w = int(parts[0]), int(parts[1])
            # 8의 배수로 정렬 (VAE 요구사항)
            h = (h // 8) * 8
            w = (w // 8) * 8
            return h, w
        except (ValueError, IndexError):
            pass

    # 파싱 실패 시 기본값
    print(f"[Resolution] Unknown format '{resolution_str}', using default {DEFAULT_HEIGHT}x{DEFAULT_WIDTH}")
    return DEFAULT_HEIGHT, DEFAULT_WIDTH


def handler(job):
    """RunPod serverless handler - CogVideoX1.5 I2V"""
    global pipe

    try:
        job_input = job["input"]

        # Load model if not loaded
        load_model()

        # Parse inputs
        image_b64 = job_input.get("image_base64")
        if not image_b64:
            return {"error": "image_base64 is required"}

        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        num_frames = int(job_input.get("num_frames", 81))
        num_inference_steps = int(job_input.get("num_inference_steps", 50))
        guidance_scale = float(job_input.get("guidance_scale", 6.0))
        seed = int(job_input.get("seed", -1))
        fps = int(job_input.get("fps", 16))
        resolution = job_input.get("resolution", "auto")

        # 해상도 결정 (CogVideoX1.5는 커스텀 해상도 지원)
        target_height, target_width = resolve_resolution(resolution)

        # Prepare image
        image = prepare_image(image_b64, target_height, target_width)

        # Seed (공식: generator는 device 미지정 = CPU)
        if seed < 0:
            seed = torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator().manual_seed(seed)

        print(
            f"[Generate] prompt='{prompt[:80]}', frames={num_frames}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}, seed={seed}, "
            f"resolution={target_width}x{target_height}, fps={fps}"
        )

        # Build pipeline kwargs (공식 cli_demo.py 기준)
        # CogVideoX1.5-5B-I2V는 height/width 전달 지원
        pipe_kwargs = {
            "image": image,
            "prompt": prompt,
            "height": target_height,
            "width": target_width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_videos_per_prompt": 1,
            "use_dynamic_cfg": True,
            "generator": generator,
        }

        # Add negative prompt only if provided
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        # Generate video
        start_time = time.time()

        with torch.no_grad():
            output = pipe(**pipe_kwargs)

        generation_time = time.time() - start_time
        print(f"[Generate] Done in {generation_time:.1f}s")

        # Export video to MP4
        from diffusers.utils import export_to_video

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_to_video(output.frames[0], tmp_path, fps=fps)

        # Read and encode to base64
        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Cleanup temp file
        os.unlink(tmp_path)

        return {
            "video_base64": video_b64,
            "generation_time": round(generation_time, 1),
            "seed": seed,
            "num_frames": num_frames,
            "resolution": f"{target_width}x{target_height}",
            "gpu_name": gpu_name,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# Start RunPod serverless
load_model()
runpod.serverless.start({"handler": handler})
