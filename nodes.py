from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .audio_builder import build_audio_timeline
from .config import QUALITY_PRESETS, StoryboardSettings, clean_output_name, comfy_input_dir, ensure_dir, parse_json_string, project_dir, safe_json_dumps
from .prompt_builder import enhance_storyboard_plan
from .scene_planner import clamp_scene_durations
from .storyboard_parser import analyze_storyboard_image, image_to_pil
from .video_assembler import assemble_movie
from .workflow_builder import export_scene_workflows, try_submit_scene_workflows


def _upscale_source_image(image: Image.Image, mode: str) -> Tuple[Image.Image, float, str]:
    normalized = str(mode or "off").lower().strip()
    factors = {"off": 1, "none": 1, "lanczos_2x": 2, "lanczos_4x": 4}
    factor = factors.get(normalized, 1)
    if factor <= 1:
        return image, 1.0, "startframe upscale disabled"
    upscaled = image.resize((image.width * factor, image.height * factor), Image.Resampling.LANCZOS)
    return upscaled, float(factor), f"startframe source upscaled with {normalized}: {image.width}x{image.height} -> {upscaled.width}x{upscaled.height}"


def _clamp_bbox(bbox: List[float], image: Image.Image, scale: float = 1.0) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in bbox]
    x0 = max(0, min(image.width - 1, x0))
    y0 = max(0, min(image.height - 1, y0))
    x1 = max(x0 + 1, min(image.width, x1))
    y1 = max(y0 + 1, min(image.height, y1))
    return x0, y0, x1, y1


def _largest_segment(mask: np.ndarray, min_len: int) -> Optional[Tuple[int, int]]:
    best: Optional[Tuple[int, int]] = None
    start: Optional[int] = None
    values = [bool(v) for v in mask.tolist()]
    for idx, active in enumerate(values + [False]):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_len and (best is None or idx - start > best[1] - best[0]):
                best = (start, idx)
            start = None
    return best


def _refine_image_window(image: Image.Image, cell_bbox: List[float], scale: float) -> Optional[Tuple[int, int, int, int]]:
    """Find the actual storyboard picture inside a designed cell.

    The fixed 4:5 board has black shot headers and white camera/music/sound
    boxes. This pass searches for the largest non-white image region and trims
    those text areas before the frame is resized for image-to-video.
    """
    cx0, cy0, cx1, cy1 = _clamp_bbox(cell_bbox, image, scale)
    if cx1 - cx0 < 48 or cy1 - cy0 < 48:
        return None
    gray = np.asarray(image.crop((cx0, cy0, cx1, cy1)).convert("L"))
    dark = gray < 245
    black = gray < 35
    row_dark = dark.mean(axis=1)
    row_black = black.mean(axis=1)
    start_scan = max(1, int(gray.shape[0] * 0.06))
    row_mask = (row_dark > 0.22) & (row_black < 0.72)
    row_mask[:start_scan] = False
    segment = _largest_segment(row_mask, max(24, int(gray.shape[0] * 0.16)))
    if not segment:
        return None

    y0, y1 = segment
    region = gray[y0:y1, :]
    col_dark = (region < 245).mean(axis=0)
    col_black = (region < 35).mean(axis=0)
    col_mask = (col_dark > 0.18) & (col_black < 0.82)
    col_segment = _largest_segment(col_mask, max(32, int(gray.shape[1] * 0.45)))
    if not col_segment:
        return None
    x0, x1 = col_segment
    pad = max(2, int(round(2 * scale)))
    return (
        max(0, cx0 + x0 + pad),
        max(0, cy0 + y0 + pad),
        min(image.width, cx0 + x1 - pad),
        min(image.height, cy0 + y1 - pad),
    )


def _scene_crop_bbox(image: Image.Image, scene: Dict[str, Any], scale: float) -> Optional[Tuple[int, int, int, int, str]]:
    cell_bbox = scene.get("source_cell_bbox")
    if cell_bbox and len(cell_bbox) == 4:
        refined = _refine_image_window(image, cell_bbox, scale)
        if refined:
            return (*refined, "refined_from_cell")
    bbox = scene.get("source_panel_bbox")
    if bbox and len(bbox) == 4:
        return (*_clamp_bbox(bbox, image, scale), "source_panel_bbox")
    return None


def _export_scene_start_frames(plan: Dict[str, Any], storyboard_image: Any, base: Path, output_name: str, width: int, height: int, upscale_mode: str = "off") -> str:
    if storyboard_image is None:
        return "No storyboard image connected to orchestrator first_frame_image; start frames were not exported."
    original = image_to_pil(storyboard_image)
    pil, bbox_scale, upscale_report = _upscale_source_image(original, upscale_mode)
    frames_dir = ensure_dir(base / "frames")
    input_root = ensure_dir(comfy_input_dir() / "storyboard2movie" / output_name)
    reports = [upscale_report]
    for idx, scene in enumerate(plan.get("scenes", []), start=1):
        crop_bbox = _scene_crop_bbox(pil, scene, bbox_scale)
        if not crop_bbox:
            reports.append(f"scene_{idx:03d}: missing source_panel_bbox")
            continue
        x0, y0, x1, y1, crop_method = crop_bbox
        crop = pil.crop((x0, y0, x1, y1)).convert("RGB")
        canvas = ImageOps.fit(crop, (int(width), int(height)), method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))
        filename = f"scene_{idx:03d}_start.png"
        output_path = frames_dir / filename
        input_path = input_root / filename
        canvas.save(output_path)
        canvas.save(input_path)
        rel_name = f"storyboard2movie/{output_name}/{filename}"
        scene["start_frame_path"] = str(output_path)
        scene["start_frame_input_name"] = rel_name
        scene["start_frame_crop_bbox"] = [x0, y0, x1, y1]
        scene["start_frame_crop_method"] = crop_method
        reports.append(f"scene_{idx:03d}: {rel_name} ({crop_method}, crop={x0},{y0},{x1},{y1})")

    plan.setdefault("project", {})["startframe_upscale_mode"] = upscale_mode
    plan.setdefault("project", {})["startframe_upscale_factor"] = bbox_scale

    ref_bbox = plan.get("project", {}).get("character_reference_bbox")
    if ref_bbox and len(ref_bbox) == 4:
        x0, y0, x1, y1 = _clamp_bbox(ref_bbox, pil, bbox_scale)
        ref = pil.crop((x0, y0, x1, y1)).convert("RGB")
        ref_out = frames_dir / "character_reference.png"
        ref_in = input_root / "character_reference.png"
        ref.save(ref_out)
        ref.save(ref_in)
        plan["project"]["character_reference_path"] = str(ref_out)
        plan["project"]["character_reference_input_name"] = f"storyboard2movie/{output_name}/character_reference.png"
    return "Exported scene start frames:\n" + "\n".join(reports)


def _render_aspect_from_scene_panels(plan: Dict[str, Any], fallback: str) -> str:
    ratios = []
    for scene in plan.get("scenes", []):
        bbox = scene.get("source_panel_bbox")
        if bbox and len(bbox) == 4:
            w = max(1, int(bbox[2]) - int(bbox[0]))
            h = max(1, int(bbox[3]) - int(bbox[1]))
            ratios.append(w / h)
    if not ratios:
        return fallback
    value = median(ratios)
    if value >= 1.45:
        return "16:9"
    if 1.15 <= value < 1.45:
        return "4:3"
    if 0.72 <= value <= 0.88:
        return "4:5"
    if 0.52 <= value <= 0.64:
        return "9:16"
    return fallback


class StoryboardImageAnalyzer:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "storyboard_image": ("IMAGE",),
                "target_language": ("STRING", {"default": "en"}),
                "fallback_style": ("STRING", {"default": "cinematic realistic"}),
                "default_duration_seconds": ("FLOAT", {"default": 12.0, "min": 0.5, "max": 300.0, "step": 0.5}),
                "default_fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "max_scenes": ("INT", {"default": 8, "min": 1, "max": 64}),
                "use_local_vlm": ("BOOLEAN", {"default": True}),
                "vlm_model_hint": ("STRING", {"default": "Qwen2.5-VL / Florence-2 / OCR fallback"}),
                "save_debug_json": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("storyboard_plan_json", "scene_count", "detected_aspect_ratio", "debug_text")
    FUNCTION = "analyze"
    CATEGORY = "Storyboard2Movie"

    def analyze(self, storyboard_image: Any, target_language: str, fallback_style: str, default_duration_seconds: float, default_fps: int, max_scenes: int, use_local_vlm: bool, vlm_model_hint: str, save_debug_json: bool) -> Tuple[str, int, str, str]:
        settings = StoryboardSettings(target_language, fallback_style, default_duration_seconds, default_fps, max_scenes, use_local_vlm, vlm_model_hint, save_debug_json)
        plan = analyze_storyboard_image(storyboard_image, settings)
        scene_count = len(plan.get("scenes", []))
        aspect = plan.get("project", {}).get("aspect_ratio", "9:16")
        debug = safe_json_dumps(plan.get("debug", {}))
        if save_debug_json:
            out = ensure_dir(project_dir("storyboard_movie") / "debug")
            (out / "last_analyzer_plan.json").write_text(safe_json_dumps(plan), encoding="utf-8")
        return safe_json_dumps(plan), scene_count, aspect, debug


class StoryboardScenePromptBuilder:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "storyboard_plan_json": ("STRING", {"multiline": True, "forceInput": True}),
                "global_style": ("STRING", {"default": "cinematic, coherent motion, realistic lighting"}),
                "prompt_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keep_character_consistency": ("BOOLEAN", {"default": True}),
                "add_camera_language": ("BOOLEAN", {"default": True}),
                "ltx_prompt_mode": (["balanced", "motion-heavy", "ugc-realistic", "cinematic", "anime", "pixel-art"], {"default": "cinematic"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_storyboard_plan_json", "prompt_list")
    FUNCTION = "build"
    CATEGORY = "Storyboard2Movie"

    def build(self, storyboard_plan_json: str, global_style: str, prompt_strength: float, keep_character_consistency: bool, add_camera_language: bool, ltx_prompt_mode: str) -> Tuple[str, str]:
        plan = parse_json_string(storyboard_plan_json)
        enhanced, prompts = enhance_storyboard_plan(plan, global_style, prompt_strength, keep_character_consistency, add_camera_language, ltx_prompt_mode)
        return safe_json_dumps(enhanced), prompts


class LTXStoryboardMovieOrchestrator:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "enhanced_storyboard_plan_json": ("STRING", {"multiline": True, "forceInput": True}),
                "output_name": ("STRING", {"default": "storyboard_movie"}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "target_width": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "quality_mode": (["4060ti_safe", "balanced", "high_quality"], {"default": "4060ti_safe"}),
                "enable_audio": ("BOOLEAN", {"default": True}),
                "enable_voiceover": ("BOOLEAN", {"default": True}),
                "enable_subtitles": ("BOOLEAN", {"default": False}),
                "enable_intermediate_exports": ("BOOLEAN", {"default": True}),
                "keep_temp_files": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "first_frame_image": ("IMAGE",),
                "startframe_upscale_mode": (["preset", "off", "lanczos_2x", "lanczos_4x"], {"default": "preset"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_video_path", "final_plan_json", "scene_video_paths_json", "audio_mix_path", "srt_path", "render_report")
    FUNCTION = "run"
    CATEGORY = "Storyboard2Movie"
    OUTPUT_NODE = True

    def run(self, enhanced_storyboard_plan_json: str, output_name: str, seed: int, fps: int, target_width: int, target_height: int, quality_mode: str, enable_audio: bool, enable_voiceover: bool, enable_subtitles: bool, enable_intermediate_exports: bool, keep_temp_files: bool, startframe_upscale_mode: str = "preset", first_frame_image: Any = None) -> Tuple[str, str, str, str, str, str]:
        plan = parse_json_string(enhanced_storyboard_plan_json)
        preset = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["4060ti_safe"])
        plan = clamp_scene_durations(plan, float(preset["max_scene_seconds"]))
        project = plan.setdefault("project", {})
        storyboard_aspect = str(project.get("aspect_ratio", "9:16"))
        aspect = _render_aspect_from_scene_panels(plan, storyboard_aspect)
        if quality_mode == "4060ti_safe" and aspect in preset.get("resolutions", {}):
            target_width, target_height = preset["resolutions"][aspect]
        elif (int(target_width), int(target_height)) == (576, 1024) and aspect in preset.get("resolutions", {}):
            target_width, target_height = preset["resolutions"][aspect]
        project["fps"] = int(fps)
        project["storyboard_aspect_ratio"] = storyboard_aspect
        project["render_aspect_ratio"] = aspect
        project["resolution"] = {"width": int(target_width), "height": int(target_height)}
        output_name = clean_output_name(output_name)
        project["quality_mode"] = quality_mode
        project["output_name"] = output_name
        if startframe_upscale_mode == "preset":
            startframe_upscale_mode = str(preset.get("startframe_upscale_mode", "off"))
        base = project_dir(output_name)
        scenes_dir = ensure_dir(base / "scenes")
        workflows_dir = ensure_dir(base / "workflows")
        audio_dir = ensure_dir(base / "audio")
        final_dir = ensure_dir(base / "final")
        frame_report = _export_scene_start_frames(plan, first_frame_image, base, output_name, int(target_width), int(target_height), startframe_upscale_mode)
        expected_scene_paths = [str(scenes_dir / f"scene_{int(s.get('id', i)):03d}.mp4") for i, s in enumerate(plan.get("scenes", []), start=1)]
        plan["expected_scene_video_paths"] = expected_scene_paths
        workflow_paths, workflow_report = export_scene_workflows(plan, workflows_dir, int(seed), int(target_width), int(target_height), int(fps))
        plan["generated_scene_workflows"] = workflow_paths
        plan_path = base / "storyboard_plan_final.json"
        plan_path.write_text(safe_json_dumps(plan), encoding="utf-8")

        audio_path = ""
        srt_path = ""
        audio_report = "Audio disabled."
        if enable_audio:
            audio_path, srt_path, audio_report = build_audio_timeline(plan, audio_dir, enable_voiceover=enable_voiceover, audio_mode="ffmpeg_placeholder")
        scene_paths_json = safe_json_dumps({"scene_video_paths": expected_scene_paths})
        final_path = final_dir / f"{output_name}_final.mp4"
        assembled, assembly_report = assemble_movie(scene_paths_json, audio_path, final_path, int(fps), int(target_width), int(target_height), "hard_cut", enable_subtitles, srt_path)
        api_report = try_submit_scene_workflows(workflow_paths)
        report = "\n\n".join([
            f"Storyboard2Movie output: {base}",
            f"Quality preset: {quality_mode} - {preset['notes']}",
            workflow_report,
            frame_report,
            f"Expected rendered scene clips:\n" + "\n".join(expected_scene_paths),
            "VideoHelperSuite suffixes such as scene_001_00001_.mp4 are accepted automatically by the assembler.",
            audio_report,
            assembly_report,
            api_report,
            "If final_video_path is empty, render the generated LTX workflows into the expected scene clip paths and rerun the assembler/orchestrator.",
        ])
        (final_dir / "render_report.txt").write_text(report, encoding="utf-8")
        return assembled, safe_json_dumps(plan), scene_paths_json, audio_path, srt_path, report


class StoryboardAudioBuilder:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "storyboard_plan_json": ("STRING", {"multiline": True, "forceInput": True}),
                "enable_music": ("BOOLEAN", {"default": True}),
                "enable_sfx": ("BOOLEAN", {"default": True}),
                "enable_voiceover": ("BOOLEAN", {"default": True}),
                "voice": ("STRING", {"default": "default"}),
                "music_style": ("STRING", {"default": "cinematic electronic"}),
                "audio_mode": (["silent", "ffmpeg_placeholder", "local_tts", "local_audio_model"], {"default": "ffmpeg_placeholder"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio_mix_path", "srt_path", "audio_report")
    FUNCTION = "build_audio"
    CATEGORY = "Storyboard2Movie"

    def build_audio(self, storyboard_plan_json: str, enable_music: bool, enable_sfx: bool, enable_voiceover: bool, voice: str, music_style: str, audio_mode: str) -> Tuple[str, str, str]:
        plan = parse_json_string(storyboard_plan_json)
        out = project_dir("storyboard_movie") / "audio"
        return build_audio_timeline(plan, out, enable_music, enable_sfx, enable_voiceover, voice, music_style, audio_mode)


class StoryboardMovieAssembler:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "scene_video_paths_json": ("STRING", {"multiline": True, "forceInput": True}),
                "output_name": ("STRING", {"default": "storyboard_movie_final"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "target_width": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "transition_mode": (["hard_cut", "crossfade", "auto"], {"default": "hard_cut"}),
                "burn_subtitles": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "audio_mix_path": ("STRING", {"default": "", "forceInput": True}),
                "srt_path": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_video_path", "assembly_report")
    FUNCTION = "assemble"
    CATEGORY = "Storyboard2Movie"
    OUTPUT_NODE = True

    def assemble(self, scene_video_paths_json: str, output_name: str, fps: int, target_width: int, target_height: int, transition_mode: str, burn_subtitles: bool, audio_mix_path: str = "", srt_path: str = "") -> Tuple[str, str]:
        final_path = project_dir("storyboard_movie") / "final" / f"{output_name}.mp4"
        return assemble_movie(scene_video_paths_json, audio_mix_path, final_path, int(fps), int(target_width), int(target_height), transition_mode, burn_subtitles, srt_path)


NODE_CLASS_MAPPINGS = {
    "StoryboardImageAnalyzer": StoryboardImageAnalyzer,
    "StoryboardScenePromptBuilder": StoryboardScenePromptBuilder,
    "LTXStoryboardMovieOrchestrator": LTXStoryboardMovieOrchestrator,
    "StoryboardAudioBuilder": StoryboardAudioBuilder,
    "StoryboardMovieAssembler": StoryboardMovieAssembler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StoryboardImageAnalyzer": "Storyboard Image Analyzer",
    "StoryboardScenePromptBuilder": "Storyboard Scene Prompt Builder",
    "LTXStoryboardMovieOrchestrator": "LTX Storyboard Movie Orchestrator",
    "StoryboardAudioBuilder": "Storyboard Audio Builder",
    "StoryboardMovieAssembler": "Storyboard Movie Assembler",
}
