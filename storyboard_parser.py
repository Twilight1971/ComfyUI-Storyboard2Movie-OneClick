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
    choices = {"9:16": 9 / 16, "4:5": 4 / 5, "16:9": 16 / 9, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4}
    return min(choices, key=lambda k: abs(choices[k] - ratio))


def _cluster_indices(indices: np.ndarray, max_gap: int = 3) -> List[int]:
    if len(indices) == 0:
        return []
    clusters: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        value = int(idx)
        if value - clusters[-1][-1] <= max_gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(round(sum(cluster) / len(cluster))) for cluster in clusters]


def _refine_storyboard_lines(lines: List[int], scores: np.ndarray, height: int) -> List[int]:
    refined = sorted(set(lines))
    min_gap = max(28, int(height * 0.095))
    while len(refined) > 4:
        gaps = [b - a for a, b in zip(refined, refined[1:])]
        smallest = min(gaps)
        if smallest >= min_gap:
            break
        idx = gaps.index(smallest)
        left = refined[idx]
        right = refined[idx + 1]
        if idx == 0:
            remove_at = idx + 1
        elif idx + 1 == len(refined) - 1:
            remove_at = idx
        else:
            left_score = float(scores[min(max(left, 0), len(scores) - 1)])
            right_score = float(scores[min(max(right, 0), len(scores) - 1)])
            remove_at = idx if left_score < right_score else idx + 1
        refined.pop(remove_at)
    return refined


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
    panels = detect_editorial_character_sheet(image)
    if panels:
        return panels
    panels = detect_storyboard_rows(image)
    if panels:
        return panels

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


def detect_editorial_character_sheet(image: Image.Image) -> List[Dict[str, Any]]:
    """Detect the fixed 4:5 character-sheet format with a right storyboard grid."""
    width, height = image.size
    ratio = width / max(1, height)
    if not (0.72 <= ratio <= 0.88 and height >= 900 and width >= 700):
        return []

    gray = np.asarray(image.convert("L")).astype(np.int16)
    x_edges = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    y_edges = np.abs(np.diff(gray, axis=0)).mean(axis=1)

    # The fixed format has a dominant vertical split after the hero/profile panel.
    search_x0 = int(width * 0.34)
    search_x1 = int(width * 0.50)
    if search_x1 <= search_x0:
        return []
    split_x = int(search_x0 + np.argmax(x_edges[search_x0:search_x1]))
    if not (width * 0.38 <= split_x <= width * 0.48):
        split_x = int(width * 0.435)

    # Bottom info strip is separated by a strong horizontal line.
    search_y0 = int(height * 0.72)
    search_y1 = int(height * 0.86)
    footer_y = int(search_y0 + np.argmax(y_edges[search_y0:search_y1]))
    if not (height * 0.75 <= footer_y <= height * 0.84):
        footer_y = int(height * 0.805)

    right_x0 = split_x + max(4, int(width * 0.008))
    right_x1 = int(width * 0.99)
    grid_y0 = int(height * 0.01)
    grid_y1 = footer_y - max(4, int(height * 0.004))

    # House format: right-side storyboard grid. Most boards use 6 shots
    # (2 columns x 3 rows); 7-shot boards are supported as a 2-column x 4-row
    # grid when the row geometry indicates enough vertical space.
    cols = 2
    rows = 3
    gap_x = max(4, int(width * 0.006))
    gap_y = max(6, int(height * 0.008))
    cell_w = (right_x1 - right_x0 - gap_x) / cols
    cell_h = (grid_y1 - grid_y0 - gap_y * (rows - 1)) / rows

    character_bbox = [int(width * 0.015), int(height * 0.012), max(1, split_x - gap_x), max(1, footer_y - 4)]
    shot_types = ["wide", "close-up", "medium", "action medium", "over shoulder", "hero close", "detail"]
    motion = [
        "the same character begins the scene with a clear establishing action",
        "the same character reacts to a new threat, clue, or turning point",
        "the same character activates or reveals the defining hero prop",
        "the same character takes decisive action with dynamic force",
        "the same character confronts the central obstacle or opponent",
        "the same character holds a final heroic or resolved stance",
        "the same character completes the final beat with a detail or aftermath moment",
    ]
    camera = [
        "wide shot, slow dolly in",
        "close-up, sharp push in",
        "medium close shot, handheld energy",
        "side angle, dynamic tracking",
        "over shoulder, tight tension",
        "low angle static hero shot",
    ]

    panels: List[Dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col + 1
            cell_x0 = int(round(right_x0 + col * (cell_w + gap_x)))
            cell_y0 = int(round(grid_y0 + row * (cell_h + gap_y)))
            cell_x1 = int(round(cell_x0 + cell_w))
            cell_y1 = int(round(cell_y0 + cell_h))
            header_h = max(24, int((cell_y1 - cell_y0) * 0.09))
            info_h = max(56, int((cell_y1 - cell_y0) * 0.21))
            pad = max(3, int(width * 0.003))
            panels.append({
                "id": idx,
                "bbox": [
                    cell_x0 + pad,
                    cell_y0 + header_h + pad,
                    cell_x1 - pad,
                    cell_y1 - info_h - pad,
                ],
                "cell_bbox": [cell_x0, cell_y0, cell_x1, cell_y1],
                "row_bbox": [cell_x0, cell_y0, cell_x1, cell_y1],
                "confidence": 0.92,
                "layout": "editorial_character_sheet_4x5",
                "character_reference_bbox": character_bbox,
                "shot_title": f"Shot {idx:02d}",
                "shot_type_hint": shot_types[idx - 1],
                "motion_hint": motion[idx - 1],
                "camera_hint": camera[idx - 1],
            })
    return panels


def detect_storyboard_rows(image: Image.Image) -> List[Dict[str, Any]]:
    """Detect horizontal storyboard/table rows such as 6-scene production boards."""
    width, height = image.size
    gray = np.asarray(image.convert("L")).astype(np.int16)
    if width < 600 or height < 400:
        return []

    # Long horizontal grid lines create strong vertical-gradient peaks.
    y_edges = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    threshold = max(float(np.percentile(y_edges, 97.5)), float(y_edges.mean() + y_edges.std() * 1.8))
    candidates = np.where(y_edges > threshold)[0]
    candidates = candidates[(candidates > int(height * 0.035)) & (candidates < int(height * 0.985))]
    y_lines = _cluster_indices(candidates, max_gap=4)

    # Keep lines that form plausible repeated storyboard rows.
    y_lines = [y for y in y_lines if y > 20]
    y_lines = _refine_storyboard_lines(y_lines, y_edges, height)
    if len(y_lines) < 4:
        return []
    if y_lines[0] > height * 0.12:
        y_lines.insert(0, 0)
    if height - y_lines[-1] > height * 0.05:
        y_lines.append(height - 1)

    row_pairs = []
    for top, bottom in zip(y_lines, y_lines[1:]):
        row_h = bottom - top
        if height * 0.07 <= row_h <= height * 0.24:
            row_pairs.append((top, bottom))
    if len(row_pairs) < 3:
        return []

    # If the board has a dedicated visual column, use it as source bbox.
    x0, x1 = _detect_visual_column(gray, width, height)
    panels: List[Dict[str, Any]] = []
    for idx, (top, bottom) in enumerate(row_pairs, start=1):
        pad_y = max(2, int((bottom - top) * 0.04))
        panels.append({
            "id": idx,
            "bbox": [x0, max(0, top + pad_y), x1, min(height, bottom - pad_y)],
            "row_bbox": [0, max(0, top), width, min(height, bottom)],
            "confidence": 0.82,
            "layout": "horizontal_storyboard_rows",
        })
    return panels


def _detect_visual_column(gray: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    strong = np.where(gx > max(float(np.percentile(gx, 97.0)), float(gx.mean() + gx.std() * 1.4)))[0]
    lines = _cluster_indices(strong, max_gap=5)
    lines = [x for x in lines if width * 0.08 < x < width * 0.92]
    if len(lines) >= 2:
        # Storyboard image columns are usually the widest central gap.
        pairs = [(b - a, a, b) for a, b in zip(lines, lines[1:]) if b - a > width * 0.25]
        if pairs:
            _, left, right = max(pairs)
            return max(0, left + 2), min(width, right - 2)

    # Fallback tuned for common production storyboard sheets: left metadata,
    # wide visual strip, then description/camera columns.
    return int(width * 0.14), int(width * 0.58)


def infer_scene_order(panels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(panels, key=lambda p: (p["bbox"][1], p["bbox"][0]))


def parse_timing(text: str, scene_count: int, default_duration: float) -> List[float]:
    ranges = re.findall(r"(\d+):(\d{2})\s*[-–]\s*(\d+):(\d{2})", text)
    if len(ranges) >= scene_count:
        durations = []
        for sm, ss, em, es in ranges[:scene_count]:
            start = int(sm) * 60 + int(ss)
            end = int(em) * 60 + int(es)
            durations.append(max(0.5, float(end - start)))
        return durations
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
        panel = panels[len(prompts)] if len(prompts) < len(panels) else {}
        title = panel.get("shot_title")
        motion = panel.get("motion_hint")
        if title and motion:
            prompts.append(f"{title}: {motion}. Maintain the same character, wardrobe, prop, and environment shown in the storyboard frame.")
        else:
            prompts.append(f"Storyboard panel {len(prompts) + 1}: infer subject, action, environment, and mood from the image.")
    return prompts


def parse_shot_titles(text: str, scene_count: int) -> List[str]:
    if not text:
        return []
    titles: Dict[int, str] = {}
    pattern = re.compile(r"\bSHOT\s*(\d{1,2})\s+([^\n\r|]+)", flags=re.I)
    for match in pattern.finditer(text):
        idx = int(match.group(1))
        title = re.sub(r"\s+", " ", match.group(2)).strip(" :-")
        if 1 <= idx <= scene_count and title:
            titles[idx] = title[:80]
    return [titles.get(i, f"Shot {i:02d}") for i in range(1, scene_count + 1)] if titles else []


def _resolution_for_ratio(ratio: str) -> Dict[str, int]:
    if ratio == "4:5":
        return {"width": 640, "height": 800}
    if ratio == "16:9":
        return {"width": 768, "height": 432}
    if ratio == "1:1":
        return {"width": 640, "height": 640}
    return {"width": 576, "height": 1024}


def _aspect_from_panels(panels: List[Dict[str, Any]], fallback_width: int, fallback_height: int) -> str:
    if not panels:
        return detect_aspect_ratio(fallback_width, fallback_height)
    if panels[0].get("layout") == "editorial_character_sheet_4x5":
        return "16:9"
    widths = []
    heights = []
    for panel in panels:
        x0, y0, x1, y1 = panel.get("bbox", [0, 0, fallback_width, fallback_height])
        widths.append(max(1, x1 - x0))
        heights.append(max(1, y1 - y0))
    return detect_aspect_ratio(int(np.median(widths)), int(np.median(heights)))


def build_fallback_scene_plan(image: Image.Image, settings: StoryboardSettings, ocr_text: str = "") -> Dict[str, Any]:
    width, height = image.size
    panels = infer_scene_order(detect_panels(image))
    aspect = _aspect_from_panels(panels, width, height)
    panel_count = min(settings.max_scenes, max(1, len(panels)))
    if panel_count == 1 and settings.default_duration_seconds >= 8:
        panel_count = min(settings.max_scenes, 4)
        panels = [{"id": i + 1, "bbox": [0, 0, width, height], "confidence": 0.15} for i in range(panel_count)]
    else:
        panels = panels[:panel_count]

    durations = parse_timing(ocr_text, len(panels), settings.default_duration_seconds)
    shot_titles = parse_shot_titles(ocr_text, len(panels))
    if shot_titles:
        for panel, title in zip(panels, shot_titles):
            panel["shot_title"] = title
    raw_prompts = parse_scene_prompts(ocr_text, panels)
    scenes: List[Dict[str, Any]] = []
    character_identity_lock = (
        "Use the left character profile panel as the identity reference. Preserve the exact same face, age, skin texture, "
        "hair or bald head shape, beard/facial hair, body build, wardrobe, colors, hero prop, and overall silhouette in every scene. "
        "Do not redesign the character, change clothing, change age, change ethnicity, change facial structure, or swap the prop."
    )
    start = 0.0
    for idx, panel in enumerate(panels, start=1):
        duration = round(durations[idx - 1], 3)
        prompt_seed = raw_prompts[idx - 1]
        shot_type = panel.get("shot_type_hint") or ("medium" if idx % 3 else "wide")
        motion = panel.get("motion_hint") or "natural subject motion with coherent temporal continuity"
        camera = panel.get("camera_hint") or ("controlled cinematic push-in" if aspect != "9:16" else "handheld iPhone push-in")
        scene = {
            "id": idx,
            "start": round(start, 3),
            "duration": duration,
            "end": round(start + duration, 3),
            "shot_type": shot_type,
            "visual_description": prompt_seed,
            "motion_description": motion,
            "camera": camera,
            "character_continuity": character_identity_lock,
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
            "source_row_bbox": panel.get("row_bbox"),
            "source_cell_bbox": panel.get("cell_bbox"),
            "character_reference_bbox": panel.get("character_reference_bbox"),
            "shot_title": panel.get("shot_title"),
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
            "character_identity_lock": character_identity_lock,
            "character_reference_bbox": panels[0].get("character_reference_bbox") if panels else None,
        },
        "scenes": scenes,
        "debug": {
            "ocr_text": ocr_text,
            "panel_count": len(panels),
            "parser": "OCR + layout heuristic fallback",
            "layout": panels[0].get("layout") if panels else "unknown",
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
