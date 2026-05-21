from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .config import DEFAULT_NEGATIVE_PROMPT, StoryboardSettings


def image_to_pil(image: Any) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor/list/PIL object to a PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    try:
        import torch  # type: ignore

        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 4:
                arr = arr[0]
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
    except Exception:
        pass
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def detect_aspect_ratio(width: int, height: int) -> str:
    ratio = width / max(1, height)
    choices = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
    return min(choices, key=lambda k: abs(choices[k] - ratio))


def extract_ocr_text(image: Image.Image) -> str:
    try:
        import pytesseract  # type: ignore

        return (pytesseract.image_to_string(image) or "").strip()
    except Exception:
        pass
    try:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(np.asarray(image))
        return "\n".join(r[1] for r in results if len(r) > 1).strip()
    except Exception:
        return ""


def detect_panels(image: Image.Image) -> List[Dict[str, Any]]:
    """Lightweight storyboard panel detector based on line/edge density."""
    width, height = image.size
    arr = np.asarray(image.convert("L"))
    panels: List[Dict[str, Any]] = []

    # Prefer simple grids, which are common for storyboards and robust without OpenCV.
    candidates: List[Tuple[int, int]] = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3)]
    best = (1, 1)
    if width > height * 1.35:
        best = (3, 1) if width > height * 2.1 else (2, 1)
    elif height > width * 1.35:
        best = (1, 3) if height > width * 2.1 else (1, 2)
    else:
        # If the image has strong central gutters, a 2x2 guess is useful.
        mid_x = arr[:, width // 2 - 2 : width // 2 + 2].mean()
        mid_y = arr[height // 2 - 2 : height // 2 + 2, :].mean()
        edge_mean = arr.mean()
        best = (2, 2) if abs(mid_x - edge_mean) > 18 or abs(mid_y - edge_mean) > 18 else (1, 1)

    cols, rows = best
    for row in range(rows):
        for col in range(cols):
            x0 = round(col * width / cols)
            y0 = round(row * height / rows)
            x1 = round((col + 1) * width / cols)
            y1 = round((row + 1) * height / rows)
            panels.append({"id": len(panels) + 1, "bbox": [x0, y0, x1, y1], "confidence": 0.45 if best != (1, 1) else 0.25})
    return panels


def infer_scene_order(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(panels, key=lambda p: (p["bbox"][1], p["bbox"][0]))


def parse_timing(text: str, scene_count: int, default_duration: float) -> List[float]:
    numbers = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", text, flags=re.I)]
    if len(numbers) >= scene_count:
        return [max(0.5, n) for n in numbers[:scene_count]]
    each = max(1.0, float(default_duration) / max(1, scene_count))
    return [each for _ in range(scene_count)]


def parse_scene_prompts(text: str, panels: List[Dict[str, Any]]) -> List[str]:
    scene_count = len(panels)
    chunks = re.split(r"(?:scene|shot|panel)\s*\d+[:\-.]?", text, flags=re.I)
    chunks = [c.strip() for c in chunks if c.strip()]
    if len(chunks) >= scene_count:
        return chunks[:scene_count]
    lines = [ln.strip(" -\t") for ln in text.splitlines() if len(ln.strip()) > 8]
    prompts = lines[:scene_count]
    while len(prompts) < scene_count:
        prompts.append(f"Storyboard panel {len(prompts) + 1}: infer subject, action, environment, and mood from the image.")
    return prompts


def _resolution_for_ratio(ratio: str) -> Dict[str, int]:
    if ratio == "16:9":
        return {"width": 768, "height": 432}
    if ratio == "1:1":
        return {"width": 640, "height": 640}
    return {"width": 576, "height": 1024}


def build_fallback_scene_plan(image: Image.Image, settings: StoryboardSettings, ocr_text: str = "") -> Dict[str, Any]:
    width, height = image.size
    aspect = detect_aspect_ratio(width, height)
    panels = infer_scene_order(detect_panels(image))
    panel_count = min(settings.max_scenes, max(1, len(panels)))
    if panel_count == 1 and settings.default_duration_seconds >= 8:
        panel_count = min(settings.max_scenes, 4)
        panels = [{"id": i + 1, "bbox": [0, 0, width, height], "confidence": 0.15} for i in range(panel_count)]
    else:
        panels = panels[:panel_count]

    durations = parse_timing(ocr_text, len(panels), settings.default_duration_seconds)
    raw_prompts = parse_scene_prompts(ocr_text, panels)
    scenes: List[Dict[str, Any]] = []
    start = 0.0
    for idx, panel in enumerate(panels, start=1):
        duration = round(durations[idx - 1], 3)
        prompt_seed = raw_prompts[idx - 1]
        scene = {
            "id": idx,
            "start": round(start, 3),
            "duration": duration,
            "end": round(start + duration, 3),
            "shot_type": "medium" if idx % 3 else "wide",
            "visual_description": prompt_seed,
            "motion_description": "natural subject motion with coherent temporal continuity",
            "camera": "controlled cinematic push-in" if aspect != "9:16" else "handheld iPhone push-in",
            "character_continuity": "preserve the same main character, clothing, props, and visual identity across scenes",
            "prompt": prompt_seed,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "audio": {
                "music_mood": "cinematic electronic",
                "sfx": ["soft transition whoosh"] if idx < len(panels) else [],
                "voiceover": "",
                "dialogue": "",
            },
            "transition_to_next": "cut" if idx < len(panels) else "none",
            "source_panel_bbox": panel.get("bbox", [0, 0, width, height]),
        }
        scenes.append(scene)
        start += duration
    return {
        "project": {
            "title": "Storyboard Movie",
            "total_duration": round(sum(s["duration"] for s in scenes), 3),
            "fps": settings.default_fps,
            "aspect_ratio": aspect,
            "resolution": _resolution_for_ratio(aspect),
            "style": f"{settings.fallback_style} TikTok UGC reel style" if aspect == "9:16" else settings.fallback_style,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        },
        "scenes": scenes,
        "debug": {
            "ocr_text": ocr_text,
            "panel_count": len(panels),
            "parser": "OCR + layout heuristic fallback",
            "vlm_warning": "No local VLM adapter is enabled in this MVP; using robust fallback parsing.",
        },
    }


def analyze_storyboard_image(image: Any, settings: StoryboardSettings) -> Dict[str, Any]:
    pil = image_to_pil(image)
    ocr = extract_ocr_text(pil)
    plan = build_fallback_scene_plan(pil, settings, ocr)
    if settings.use_local_vlm:
        plan.setdefault("debug", {})["vlm_adapter"] = (
            f"Requested local VLM ({settings.vlm_model_hint}); no mandatory adapter loaded. "
            "Install/implement Qwen2.5-VL, Florence-2, or Moondream adapter to replace this fallback."
        )
    return plan
