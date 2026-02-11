"""
RunPod Serverless Handler - Wan2.2-I2V-A14B (Image-to-Video)
Wan 2.2 MoE I2V model with Lightning LoRA acceleration + optional NSFW LoRA

Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers
Architecture: MoE (Mixture-of-Experts) - dual transformer (high noise + low noise)
LoRA: lightx2v/Wan2.2-Lightning (4-8 steps acceleration)
Resolution: Flexible (480x832 default, supports portrait/landscape/square)
Frames: 81 (5s@16fps)
VRAM: ~56GB active (A100 80GB recommended)

References:
  - https://github.com/AleefBilal/wan22-I2V-runpod
  - https://huggingface.co/spaces/obsxrver/Wan2.2-I2V-LoRA-Demo
"""

import os
import io
import gc
import time
import base64
import tempfile
import traceback

import torch
import numpy as np
from PIL import Image

import runpod

# ===========================================================================
# Configuration
# ===========================================================================

# Model paths on Network Volume (pre-downloaded)
MODEL_DIR = "/runpod-volume/models/Wan2.2-I2V-A14B-Diffusers"
LORA_DIR = "/runpod-volume/models/lora"

# Lightning LoRA for speed acceleration (4-8 steps instead of 40-50)
LIGHTNING_LORA_DIR = os.path.join(LORA_DIR, "Wan2.2-Lightning")
LIGHTNING_LORA_SUBDIR = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1"
LIGHTNING_HIGH_NOISE = "high_noise_model.safetensors"
LIGHTNING_LOW_NOISE = "low_noise_model.safetensors"

# NSFW LoRA (optional - only loaded if present on disk)
NSFW_LORA_DIR = os.path.join(LORA_DIR, "nsfw-lora")
NSFW_HIGH_NOISE = "NSFW-22-H-e8.safetensors"
NSFW_LOW_NOISE = "NSFW-22-L-e8.safetensors"

# Cache version marker
CACHE_VERSION = "wan22-i2v-a14b-v1"

# Image sizing
MAX_DIM = 832
MIN_DIM = 480
MULTIPLE_OF = 16

# Video defaults
FIXED_FPS = 16
MIN_FRAMES = 8
MAX_FRAMES = 161  # ~10s @ 16fps

# Wan2.2 I2V supports flexible resolutions (height, width)
SUPPORTED_RESOLUTIONS = {
    "16:9": (480, 832),      # landscape wide (default, fast)
    "9:16": (832, 480),      # portrait tall (fast)
    "4:3": (480, 640),       # landscape standard
    "3:4": (640, 480),       # portrait standard
    "1:1": (480, 480),       # square
    "16:9-HD": (720, 1280),  # HD landscape (slower, higher quality)
    "9:16-HD": (1280, 720),  # HD portrait (slower, higher quality)
    "1:1-HD": (720, 720),    # HD square (slower, higher quality)
}

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, motion artifacts, unstable motion, jitter, "
    "frame jitter, wobbling limbs, motion distortion, inconsistent movement, "
    "robotic movement, animation-like motion, awkward transitions, "
    "incorrect body mechanics, unnatural posing, off-balance poses, "
    "frozen frames, duplicated frames, frame skipping, warped motion, "
    "bad anatomy, incorrect proportions, deformed body, twisted torso, "
    "broken joints, dislocated limbs, distorted neck, malformed hands, "
    "extra fingers, missing fingers, fused fingers, extra limbs, "
    "blurry details, ghosting, compression noise, jpeg artifacts"
)

# ===========================================================================
# Global state
# ===========================================================================
pipe = None
gpu_name = "Unknown"


# ===========================================================================
# Image utilities
# ===========================================================================
def resize_image(image: Image.Image) -> Image.Image:
    """Resize image to model-compatible dimensions.

    Maintains aspect ratio, clamps to MAX_DIM/MIN_DIM,
    and ensures dimensions are multiples of MULTIPLE_OF.
    """
    w, h = image.size

    # Scale down if larger than MAX_DIM
    scale = min(MAX_DIM / max(w, h), 1.0)
    w, h = int(w * scale), int(h * scale)

    # Snap to multiples of 16
    w = (w // MULTIPLE_OF) * MULTIPLE_OF
    h = (h // MULTIPLE_OF) * MULTIPLE_OF

    # Enforce minimum
    w = max(MIN_DIM, w)
    h = max(MIN_DIM, h)

    return image.resize((w, h), Image.Resampling.LANCZOS)


def detect_best_resolution(orig_w, orig_h):
    """Detect best resolution based on input image aspect ratio."""
    aspect = orig_w / orig_h

    if aspect > 1.5:
        res = SUPPORTED_RESOLUTIONS["16:9"]
        name = "16:9"
    elif aspect > 1.15:
        res = SUPPORTED_RESOLUTIONS["4:3"]
        name = "4:3"
    elif aspect > 0.85:
        res = SUPPORTED_RESOLUTIONS["1:1"]
        name = "1:1"
    elif aspect > 0.65:
        res = SUPPORTED_RESOLUTIONS["3:4"]
        name = "3:4"
    else:
        res = SUPPORTED_RESOLUTIONS["9:16"]
        name = "9:16"

    h, w = res
    print(f"[Resolution] Auto-detected: {name} ({w}x{h}) for input {orig_w}x{orig_h} (aspect={aspect:.2f})")
    return h, w


def resolve_resolution(resolution_str):
    """Convert resolution string to (height, width) tuple."""
    if not resolution_str or resolution_str == "auto":
        return None, None

    if resolution_str in SUPPORTED_RESOLUTIONS:
        return SUPPORTED_RESOLUTIONS[resolution_str]

    # Parse "HxW" format
    if "x" in resolution_str:
        try:
            parts = resolution_str.lower().split("x")
            h, w = int(parts[0]), int(parts[1])
            h = max((h // MULTIPLE_OF) * MULTIPLE_OF, MIN_DIM)
            w = max((w // MULTIPLE_OF) * MULTIPLE_OF, MIN_DIM)
            h = min(h, 1280)
            w = min(w, 1280)
            print(f"[Resolution] Custom: {w}x{h} (validated)")
            return h, w
        except (ValueError, IndexError):
            pass

    print(f"[Resolution] Unknown format '{resolution_str}', using auto-detect")
    return None, None


def prepare_image(image_b64, target_height=None, target_width=None):
    """Decode base64 image and resize to target resolution."""
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    orig_w, orig_h = image.size
    print(f"[Image] Original size: {orig_w}x{orig_h}")

    if target_height is None or target_width is None:
        target_height, target_width = detect_best_resolution(orig_w, orig_h)

    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"[Image] Resized to: {target_width}x{target_height}")

    return image, target_height, target_width


# ===========================================================================
# Cache management (Network Volume)
# ===========================================================================
def get_cache_marker_path():
    """Cache version marker file path."""
    return os.path.join(os.path.dirname(MODEL_DIR), ".cache_version")


def check_cache_version():
    """Check if cache version matches expected version."""
    marker_path = get_cache_marker_path()
    if not os.path.exists(marker_path):
        return False
    with open(marker_path, "r") as f:
        stored = f.read().strip()
    return stored == CACHE_VERSION


def mark_cache_valid():
    """Write cache version marker after successful model load."""
    marker_path = get_cache_marker_path()
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        f.write(CACHE_VERSION)
    print(f"[Cache] Version marker written: {CACHE_VERSION}")


def verify_model_files():
    """Verify that model files exist on Network Volume."""
    required_dirs = ["transformer", "transformer_2", "text_encoder", "vae"]

    if not os.path.exists(MODEL_DIR):
        print(f"[Model] ERROR: Model directory not found: {MODEL_DIR}")
        print(f"[Model] Please pre-download the model to the Network Volume.")
        print(f"[Model] Run: huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir {MODEL_DIR}")
        return False

    for subdir in required_dirs:
        path = os.path.join(MODEL_DIR, subdir)
        if not os.path.exists(path):
            print(f"[Model] ERROR: Required directory missing: {path}")
            return False

    # Check model_index.json
    index_path = os.path.join(MODEL_DIR, "model_index.json")
    if not os.path.exists(index_path):
        print(f"[Model] ERROR: model_index.json not found in {MODEL_DIR}")
        return False

    print(f"[Model] All required model files verified at {MODEL_DIR}")
    return True


def verify_lora_files():
    """Verify Lightning LoRA files exist. Returns (has_lightning, has_nsfw)."""
    has_lightning = False
    has_nsfw = False

    # Check Lightning LoRA
    lightning_high = os.path.join(LIGHTNING_LORA_DIR, LIGHTNING_LORA_SUBDIR, LIGHTNING_HIGH_NOISE)
    lightning_low = os.path.join(LIGHTNING_LORA_DIR, LIGHTNING_LORA_SUBDIR, LIGHTNING_LOW_NOISE)

    if os.path.exists(lightning_high) and os.path.exists(lightning_low):
        has_lightning = True
        print(f"[LoRA] Lightning LoRA found: {LIGHTNING_LORA_DIR}")
    else:
        print(f"[LoRA] WARNING: Lightning LoRA not found at {LIGHTNING_LORA_DIR}")
        print(f"[LoRA] Will run without acceleration (slower, needs more steps)")

    # Check NSFW LoRA (optional)
    nsfw_high = os.path.join(NSFW_LORA_DIR, NSFW_HIGH_NOISE)
    nsfw_low = os.path.join(NSFW_LORA_DIR, NSFW_LOW_NOISE)

    if os.path.exists(nsfw_high) and os.path.exists(nsfw_low):
        has_nsfw = True
        print(f"[LoRA] NSFW LoRA found: {NSFW_LORA_DIR}")
        print(f"[LoRA]   High: {NSFW_HIGH_NOISE}")
        print(f"[LoRA]   Low:  {NSFW_LOW_NOISE}")
    else:
        print(f"[LoRA] NSFW LoRA not found (optional, skipping)")

    return has_lightning, has_nsfw


# ===========================================================================
# Model loading
# ===========================================================================
def load_model():
    """Load Wan2.2-I2V-A14B pipeline with LoRAs."""
    global pipe, gpu_name

    if pipe is not None:
        return

    print("=" * 60)
    print("[Model] Loading Wan2.2-I2V-A14B-Diffusers...")
    print(f"[Model] Cache version: {CACHE_VERSION}")
    print("=" * 60)
    start = time.time()

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU] {gpu_name} ({vram_gb:.1f} GB)")
    else:
        raise RuntimeError("CUDA not available! This model requires a GPU.")

    # Step 1: Verify model files
    if not verify_model_files():
        raise RuntimeError(
            f"Model files not found at {MODEL_DIR}. "
            "Please pre-download the model to the RunPod Network Volume. "
            "See MIGRATION_PLAN.md for instructions."
        )

    has_lightning, has_nsfw = verify_lora_files()

    # Step 2: Load pipeline with MoE dual transformers
    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel

    print("[Model] Loading transformer (high noise expert)...")
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print("[Model] Loading transformer_2 (low noise expert)...")
    transformer_2 = WanTransformer3DModel.from_pretrained(
        MODEL_DIR,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
    )

    print("[Model] Loading full pipeline...")
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        transformer=transformer,
        transformer_2=transformer_2,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Step 3: Load LoRAs (Lightning + NSFW)
    adapter_names_high = []  # adapters for transformer (high noise)
    adapter_names_low = []   # adapters for transformer_2 (low noise)

    # 3a: Lightning LoRA (speed acceleration)
    if has_lightning:
        print("[LoRA] Loading Lightning LoRA (high noise)...")
        lora_subpath = os.path.join(LIGHTNING_LORA_DIR, LIGHTNING_LORA_SUBDIR)
        pipe.load_lora_weights(
            lora_subpath,
            weight_name=LIGHTNING_HIGH_NOISE,
            adapter_name="lightning_high",
        )
        adapter_names_high.append("lightning_high")

        print("[LoRA] Loading Lightning LoRA (low noise)...")
        pipe.load_lora_weights(
            lora_subpath,
            weight_name=LIGHTNING_LOW_NOISE,
            adapter_name="lightning_low",
            load_into_transformer_2=True,
        )
        adapter_names_low.append("lightning_low")

    # 3b: NSFW LoRA (content freedom)
    if has_nsfw:
        print("[LoRA] Loading NSFW LoRA (high noise)...")
        pipe.load_lora_weights(
            NSFW_LORA_DIR,
            weight_name=NSFW_HIGH_NOISE,
            adapter_name="nsfw_high",
        )
        adapter_names_high.append("nsfw_high")

        print("[LoRA] Loading NSFW LoRA (low noise)...")
        pipe.load_lora_weights(
            NSFW_LORA_DIR,
            weight_name=NSFW_LOW_NOISE,
            adapter_name="nsfw_low",
            load_into_transformer_2=True,
        )
        adapter_names_low.append("nsfw_low")

    # 3c: Fuse all LoRAs into model weights (zero overhead at inference)
    all_adapters = adapter_names_high + adapter_names_low
    if all_adapters:
        print(f"[LoRA] Fusing LoRAs: {all_adapters}")
        all_weights = [1.0] * len(all_adapters)
        pipe.set_adapters(all_adapters, adapter_weights=all_weights)

        if adapter_names_high:
            pipe.fuse_lora(
                adapter_names=adapter_names_high,
                lora_scale=1.0,
                components=["transformer"],
            )
        if adapter_names_low:
            pipe.fuse_lora(
                adapter_names=adapter_names_low,
                lora_scale=1.0,
                components=["transformer_2"],
            )
        pipe.unload_lora_weights()
        print(f"[LoRA] All LoRAs fused and unloaded (zero overhead)")

    # Step 4: Text encoder quantization - DISABLED
    # torchao is incompatible with torch 2.7.1 and causes diffusers import crash
    # 80GB VRAM is sufficient without quantization
    print("[Quantize] Skipped (80GB VRAM sufficient without quantization)")

    # Step 5: VAE optimizations
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        if hasattr(pipe.vae, 'enable_tiling'):
            pipe.vae.enable_tiling()
            print("[VAE] Tiling enabled")
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
            print("[VAE] Slicing enabled")

    # Step 6: Ensure bfloat16 precision on transformers
    pipe.transformer.to(torch.bfloat16)
    pipe.transformer_2.to(torch.bfloat16)

    elapsed = time.time() - start
    print("=" * 60)
    print(f"[Model] Loaded in {elapsed:.1f}s")
    print(f"[Model] Lightning LoRA: {'enabled' if has_lightning else 'disabled'}")
    print(f"[Model] Ready for inference!")
    print("=" * 60)

    # Mark cache valid
    mark_cache_valid()


# ===========================================================================
# RunPod handler
# ===========================================================================
def handler(job):
    """RunPod serverless handler - Wan2.2 I2V."""
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
        negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        num_frames = int(job_input.get("num_frames", 81))
        num_inference_steps = int(job_input.get("num_inference_steps", 6))
        guidance_scale = float(job_input.get("guidance_scale", 1.0))
        guidance_scale_2 = float(job_input.get("guidance_scale_2", 1.0))
        seed = int(job_input.get("seed", -1))
        fps = int(job_input.get("fps", FIXED_FPS))
        resolution = job_input.get("resolution", "auto")

        # Validate num_frames (clamp to valid range)
        num_frames = max(MIN_FRAMES, min(MAX_FRAMES, num_frames))
        # Wan 2.2 frame rule: 4N+1
        valid_frames = [4 * n + 1 for n in range(1, 41)]  # 5 to 161
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
                image, target_height, target_width = prepare_image(
                    image_b64, target_height, target_width
                )

        # Seed
        if seed < 0:
            seed = torch.randint(0, 2**31, (1,)).item()
        generator = torch.Generator(device="cuda").manual_seed(seed)

        print(
            f"[Generate] prompt='{prompt[:80]}...', frames={num_frames}, "
            f"steps={num_inference_steps}, guidance={guidance_scale}/{guidance_scale_2}, "
            f"seed={seed}, resolution={target_width}x{target_height}, fps={fps}"
        )

        # Clear GPU cache before generation
        torch.cuda.empty_cache()
        gc.collect()

        # Generate video
        start_time = time.time()

        with torch.no_grad():
            output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=target_height,
                width=target_width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                generator=generator,
            )

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

        # Cleanup
        os.unlink(tmp_path)
        del output
        torch.cuda.empty_cache()

        return {
            "video_base64": video_b64,
            "generation_time": round(generation_time, 1),
            "seed": seed,
            "num_frames": num_frames,
            "resolution": f"{target_width}x{target_height}",
            "gpu_name": gpu_name,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
        }

    except Exception as e:
        traceback.print_exc()
        torch.cuda.empty_cache()
        return {"error": str(e)}


# ===========================================================================
# Entry point
# ===========================================================================
print("[Startup] Wan2.2-I2V-A14B RunPod Handler starting...")
load_model()
runpod.serverless.start({"handler": handler})
