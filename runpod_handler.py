"""
RunPod Serverless Handler - Wan2.1-Fun-V1.3-InP (Image-to-Video)
VideoX-Fun InPainting-based I2V model (Alibaba PAI)

Model: alibaba-pai/Wan2.1-Fun-V1.3-InP
Base: Wan 2.1 14B (no safety alignment - uncensored)
Resolution: Flexible (480x832 default, supports portrait/landscape/square)
Frames: 81 (5s@16fps)
VRAM: ~30GB (A100 80GB recommended)
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
import numpy as np
from PIL import Image

import runpod

# --- Configuration ---
MODEL_ID = "alibaba-pai/Wan2.1-Fun-V1.3-InP"
CACHE_DIR = "/runpod-volume/huggingface"

# Cache version: change this to force re-download on Network Volume
CACHE_VERSION = "wan21-fun-v1.3-inp-v1"

# Safetensors minimum size (incomplete downloads are smaller)
SAFETENSORS_MIN_SIZE_MB = 10

# Default resolution (Wan 2.1 standard 720P landscape)
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832

# Wan2.1-Fun supports flexible resolutions (height, width)
# Unlike CogVideoX, portrait is fully supported!
SUPPORTED_RESOLUTIONS = {
    "16:9": (480, 832),     # landscape wide (default, fast)
    "9:16": (832, 480),     # portrait tall (fast)
    "4:3": (480, 640),      # landscape standard
    "3:4": (640, 480),      # portrait standard
    "1:1": (480, 480),      # square
    "16:9-HD": (720, 1280), # HD landscape (slower, higher quality)
    "9:16-HD": (1280, 720), # HD portrait (slower, higher quality)
    "1:1-HD": (720, 720),   # HD square (slower, higher quality)
}

pipe = None
gpu_name = "Unknown"


def get_cache_marker_path(cache_dir):
    """Cache version marker file path"""
    return os.path.join(cache_dir, ".cache_version")


def needs_force_clean(cache_dir):
    """Check version marker to determine if force re-download is needed"""
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
    """Write version marker after successful model load"""
    marker_path = get_cache_marker_path(cache_dir)
    with open(marker_path, "w") as f:
        f.write(CACHE_VERSION)
    print(f"[Cache] Version marker written: {CACHE_VERSION}")


def force_clean_model_cache(cache_dir, model_id):
    """Completely remove model cache for clean re-download"""
    model_cache_path = os.path.join(
        cache_dir, "models--" + model_id.replace("/", "--")
    )

    if os.path.exists(model_cache_path):
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

    # Also remove version marker
    marker_path = get_cache_marker_path(cache_dir)
    if os.path.exists(marker_path):
        os.remove(marker_path)


def validate_cache(cache_dir, model_id):
    """Validate integrity of cached model files on Network Volume"""
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

            # JSON file validation
            if f.endswith(".json"):
                try:
                    with open(filepath, "r") as fh:
                        json.load(fh)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    print(f"[Cache] CORRUPTED JSON: {filepath}")
                    corrupted = True
                    break

            # Safetensors validation
            elif f.endswith(".safetensors"):
                file_size = os.path.getsize(filepath)
                file_size_mb = file_size / (1024 * 1024)
                safetensors_files.append((f, file_size_mb))
                total_safetensors_size += file_size

                if file_size_mb < SAFETENSORS_MIN_SIZE_MB:
                    print(f"[Cache] CORRUPTED safetensors (too small: {file_size_mb:.1f}MB): {filepath}")
                    corrupted = True
                    break

            # Binary model files
            elif f.endswith((".model", ".bin")):
                file_size = os.path.getsize(filepath)
                if file_size < 1024:
                    print(f"[Cache] CORRUPTED binary (too small: {file_size}B): {filepath}")
                    corrupted = True
                    break

        if corrupted:
            break

    # Total safetensors size check (Wan 2.1 14B ~ 28GB+ in safetensors)
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
        marker_path = get_cache_marker_path(cache_dir)
        if os.path.exists(marker_path):
            os.remove(marker_path)
        print(f"[Cache] Removed: {model_cache_path}")
        return False
    else:
        print(f"[Cache] All validations passed!")
        return True


def cleanup_old_models(cache_dir, current_model_id):
    """Remove old model caches to free disk space"""
    if not os.path.exists(cache_dir):
        return

    current_cache_name = "models--" + current_model_id.replace("/", "--")

    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if item.startswith("models--") and item != current_cache_name and os.path.isdir(item_path):
            print(f"[Cleanup] Removing old model cache: {item}")
            shutil.rmtree(item_path, ignore_errors=True)
            print(f"[Cleanup] Removed: {item}")

    # Clean orphaned temp/lock files
    for item in os.listdir(cache_dir):
        item_path = os.path.join(cache_dir, item)
        if item.endswith(".lock") or item.endswith(".tmp"):
            try:
                os.remove(item_path)
                print(f"[Cleanup] Removed temp file: {item}")
            except OSError:
                pass


def load_model():
    """Load Wan2.1-Fun-V1.3-InP model"""
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

    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Step 1: Clean up old models (CogVideoX, etc.)
    cleanup_old_models(CACHE_DIR, MODEL_ID)

    # Step 2: Check version marker → force clean if mismatch
    if needs_force_clean(CACHE_DIR):
        print(f"[Cache] === FORCE CLEAN MODE ===")
        force_clean_model_cache(CACHE_DIR, MODEL_ID)
    else:
        validate_cache(CACHE_DIR, MODEL_ID)

    # Step 3: Check disk space
    disk_stat = shutil.disk_usage(CACHE_DIR)
    free_gb = disk_stat.free / (1024**3)
    total_gb = disk_stat.total / (1024**3)
    print(f"[Disk] Free: {free_gb:.1f}GB / Total: {total_gb:.1f}GB")
    if free_gb < 5.0:
        print(f"[Disk] WARNING: Low disk space! Model download may fail (need ~37GB).")

    # Step 4: Load pipeline
    # Try diffusers auto-detect first (works if diffusers has WanFunInpaintPipeline)
    pipeline_loaded = False

    # Approach 1: Auto-detect from model_index.json
    try:
        from diffusers import DiffusionPipeline
        print(f"[Model] Trying DiffusionPipeline.from_pretrained (auto-detect)...")
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        )
        pipeline_loaded = True
        print(f"[Model] Loaded via DiffusionPipeline auto-detect: {type(pipe).__name__}")
    except Exception as e1:
        print(f"[Model] Auto-detect failed: {e1}")

        # Approach 2: Explicit WanFunInpaintPipeline
        try:
            from diffusers import WanFunInpaintPipeline
            print(f"[Model] Trying WanFunInpaintPipeline explicitly...")
            pipe = WanFunInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                cache_dir=CACHE_DIR,
            )
            pipeline_loaded = True
            print(f"[Model] Loaded via WanFunInpaintPipeline")
        except Exception as e2:
            print(f"[Model] WanFunInpaintPipeline failed: {e2}")

            # Approach 3: Use VideoX-Fun's pipeline (if installed)
            try:
                import sys
                # Try importing from videox_fun package
                from videox_fun.pipelines.pipeline_wan_fun_inpaint import WanFunInpaintPipeline as VXPipeline
                print(f"[Model] Trying VideoX-Fun pipeline...")
                pipe = VXPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.bfloat16,
                    cache_dir=CACHE_DIR,
                )
                pipeline_loaded = True
                print(f"[Model] Loaded via VideoX-Fun pipeline")
            except Exception as e3:
                print(f"[Model] VideoX-Fun pipeline failed: {e3}")
                raise RuntimeError(
                    f"Failed to load model with all approaches:\n"
                    f"  1) DiffusionPipeline: {e1}\n"
                    f"  2) WanFunInpaintPipeline: {e2}\n"
                    f"  3) VideoX-Fun: {e3}"
                )

    # Enable VAE optimizations if available
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
            print(f"[Model] VAE tiling enabled")
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
            print(f"[Model] VAE slicing enabled")

    # GPU placement
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

    # Step 5: Mark cache valid
    mark_cache_valid(CACHE_DIR)
    print(f"[Model] Ready for inference!")


def detect_best_resolution(orig_w, orig_h):
    """Detect best resolution based on input image aspect ratio

    Wan2.1-Fun supports flexible resolutions including portrait!
    """
    aspect = orig_w / orig_h

    if aspect > 1.5:
        # Wide landscape (16:9)
        res = SUPPORTED_RESOLUTIONS["16:9"]
        name = "16:9"
    elif aspect > 1.15:
        # Standard landscape (4:3)
        res = SUPPORTED_RESOLUTIONS["4:3"]
        name = "4:3"
    elif aspect > 0.85:
        # Square-ish (1:1)
        res = SUPPORTED_RESOLUTIONS["1:1"]
        name = "1:1"
    elif aspect > 0.65:
        # Standard portrait (3:4)
        res = SUPPORTED_RESOLUTIONS["3:4"]
        name = "3:4"
    else:
        # Tall portrait (9:16)
        res = SUPPORTED_RESOLUTIONS["9:16"]
        name = "9:16"

    h, w = res
    print(f"[Resolution] Auto-detected: {name} ({w}x{h}) for input {orig_w}x{orig_h} (aspect={aspect:.2f})")
    return h, w


def prepare_image(image_b64, target_height=None, target_width=None):
    """Decode base64 image and resize to target resolution"""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    orig_w, orig_h = image.size
    print(f"[Image] Original size: {orig_w}x{orig_h}")

    # Auto-detect resolution if not specified
    if target_height is None or target_width is None:
        target_height, target_width = detect_best_resolution(orig_w, orig_h)

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"[Image] Resized to: {target_width}x{target_height}")

    return image, target_height, target_width


def resolve_resolution(resolution_str):
    """Convert resolution string to (height, width) tuple"""
    if not resolution_str or resolution_str == "auto":
        return None, None  # Will be auto-detected from image

    # Lookup in SUPPORTED_RESOLUTIONS
    if resolution_str in SUPPORTED_RESOLUTIONS:
        return SUPPORTED_RESOLUTIONS[resolution_str]

    # Parse "HxW" format
    if "x" in resolution_str:
        try:
            parts = resolution_str.lower().split("x")
            h, w = int(parts[0]), int(parts[1])

            # Ensure dimensions are multiples of 16
            h = max((h // 16) * 16, 256)
            w = max((w // 16) * 16, 256)

            # Cap at reasonable size
            h = min(h, 1280)
            w = min(w, 1280)

            print(f"[Resolution] Custom: {w}x{h} (validated)")
            return h, w
        except (ValueError, IndexError):
            pass

    # Fallback
    print(f"[Resolution] Unknown format '{resolution_str}', using default {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}")
    return DEFAULT_HEIGHT, DEFAULT_WIDTH


def handler(job):
    """RunPod serverless handler - Wan2.1-Fun I2V"""
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

        # Wan 2.1 frame rule: 4N+1
        # Valid values: 5, 9, 13, 17, 21, 25, ..., 77, 81, 85
        valid_frames = [4 * n + 1 for n in range(1, 22)]  # 5 to 85
        if num_frames not in valid_frames:
            num_frames = min(valid_frames, key=lambda x: abs(x - num_frames))
            print(f"[Frames] Adjusted to nearest valid value (4N+1): {num_frames}")

        # Resolve resolution
        if resolution == "auto":
            image, target_height, target_width = prepare_image(image_b64)
        else:
            target_height, target_width = resolve_resolution(resolution)
            if target_height is None:
                image, target_height, target_width = prepare_image(image_b64)
            else:
                image, target_height, target_width = prepare_image(image_b64, target_height, target_width)

        # Seed
        if seed < 0:
            seed = torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator().manual_seed(seed)

        print(
            f"[Generate] prompt='{prompt[:80]}', frames={num_frames}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}, seed={seed}, "
            f"resolution={target_width}x{target_height}, fps={fps}"
        )

        # Build pipeline kwargs
        pipe_kwargs = {
            "image": image,
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "height": target_height,
            "width": target_width,
        }

        # Add negative prompt if provided
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
