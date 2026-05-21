from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PACKAGE_NAME = "ComfyUI-Storyboard2Movie-OneClick"
MODELS_ROOT = Path("N:/KI_Daten/models")
LTX_I2V_TEMPLATE = Path("N:/KI_Daten/custom_nodes/ComfyUI-LTXVideo/example_workflows/LTX-2_I2V_Distilled_wLora.json")
COMFY_INPUT_FALLBACK = Path("N:/pinokio/api/comfy.git/app/input")
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, flicker, jitter, distorted face, warped hands, "
    "duplicated limbs, bad anatomy, inconsistent character, random text, "
    "logo artifacts, compression artifacts, broken motion, scene jump"
)


def _default_output_root() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_output_directory()) / "storyboard_movie"
    except Exception:
        return Path.cwd() / "outputs" / "storyboard_movie"


OUTPUT_ROOT = _default_output_root()


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    "4060ti_safe": {
        "max_scene_seconds": 4.0,
        "fps": 24,
        "resolutions": {"9:16": (576, 1024), "4:5": (640, 800), "16:9": (768, 432), "1:1": (640, 640)},
        "notes": "16GB VRAM safe: short per-scene generations, fp16/bf16, offload where supported.",
    },
    "balanced": {
        "max_scene_seconds": 5.0,
        "fps": 24,
        "resolutions": {"9:16": (720, 1280), "4:5": (720, 900), "16:9": (960, 544), "1:1": (768, 768)},
        "notes": "Higher resolution if VRAM allows; keep scenes short.",
    },
    "high_quality": {
        "max_scene_seconds": 6.0,
        "fps": 24,
        "resolutions": {"9:16": (720, 1280), "4:5": (960, 1200), "16:9": (1280, 720), "1:1": (1024, 1024)},
        "notes": "May exceed 16GB VRAM depending on installed LTXVideo settings.",
    },
}


@dataclass
class StoryboardSettings:
    target_language: str = "en"
    fallback_style: str = "cinematic realistic"
    default_duration_seconds: float = 12.0
    default_fps: int = 24
    max_scenes: int = 8
    use_local_vlm: bool = True
    vlm_model_hint: str = "Qwen2.5-VL / Florence-2 / OCR fallback"
    save_debug_json: bool = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json_dumps(data: Any, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)


def parse_json_string(value: str) -> Dict[str, Any]:
    if not value or not value.strip():
        raise ValueError("Expected a non-empty storyboard JSON string.")
    return json.loads(value)


def project_dir(output_name: str) -> Path:
    cleaned = "".join(c for c in output_name.strip() if c.isalnum() or c in ("-", "_")).strip("_-")
    cleaned = cleaned or "storyboard_movie"
    return ensure_dir(OUTPUT_ROOT / cleaned)


def load_user_ltx_mapping() -> Dict[str, Any]:
    candidates = [
        Path(__file__).with_name("ltx_node_mapping.json"),
        Path(os.environ.get("STORYBOARD2MOVIE_LTX_MAPPING", "")),
    ]
    for candidate in candidates:
        if candidate and str(candidate) != "." and candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {
        "template_mode": "ltxvideo_i2v_distilled_template",
        "template_path": str(LTX_I2V_TEMPLATE),
        "models_root": str(MODELS_ROOT),
        "checkpoint": "LTX23/ltx-2.3-22b-dev-fp8.safetensors",
        "audio_vae": "LTX23/ltx-2.3-22b-distilled_audio_vae.safetensors",
        "text_encoder_mode": "checkpoint_clip",
        "gemma_text_encoder": "gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors",
        "gemma_connector": "LTX23\\ltx-2.3-22b-dev-fp8.safetensors",
        "gemma_max_length": 512,
        "latent_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "notes": "Generated scene workflows are patched from the installed ComfyUI-LTXVideo I2V distilled template. 4060ti_safe uses checkpoint_clip text encoding by default to avoid Gemma 12B VRAM OOM.",
        "recommended_server": "http://127.0.0.1:666",
    }


def comfy_input_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_input_directory())
    except Exception:
        return COMFY_INPUT_FALLBACK
