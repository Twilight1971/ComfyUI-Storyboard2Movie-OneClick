from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from .config import ensure_dir
from .ffmpeg_utils import burn_subtitles, check_ffmpeg_available, final_mux


def parse_scene_paths(scene_video_paths_json: str) -> List[str]:
    data = json.loads(scene_video_paths_json)
    if isinstance(data, list):
        return [str(p) for p in data]
    return [str(p) for p in data.get("scene_video_paths", [])]


def _resolve_rendered_scene(path: str) -> str | None:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    matches = sorted(candidate.parent.glob(f"{candidate.stem}_*.mp4"))
    if matches:
        return str(matches[-1])
    return None


def assemble_movie(scene_video_paths_json: str, audio_mix_path: str, output_path: str | Path, fps: int, target_width: int, target_height: int, transition_mode: str = "hard_cut", burn_srt: bool = False, srt_path: str = "") -> Tuple[str, str]:
    ok, ff = check_ffmpeg_available()
    if not ok:
        return "", ff
    paths = parse_scene_paths(scene_video_paths_json)
    existing = [resolved for p in paths if (resolved := _resolve_rendered_scene(p))]
    missing = [p for p in paths if not _resolve_rendered_scene(p)]
    if not existing:
        return "", "No rendered scene clips were found. Render the generated LTX scene workflows first, then run the assembler."
    ensure_dir(Path(output_path).parent)
    reports = [ff]
    if missing:
        reports.append("Missing scene clips skipped:\n" + "\n".join(missing))
    if transition_mode != "hard_cut":
        reports.append(f"transition_mode={transition_mode} requested; MVP uses normalized hard cuts for reliability on Windows.")
    ok, report = final_mux(existing, audio_mix_path if audio_mix_path and Path(audio_mix_path).exists() else None, output_path, {"fps": fps, "width": target_width, "height": target_height})
    reports.append(report)
    final_path = str(output_path) if ok else ""
    if ok and burn_srt and srt_path and Path(srt_path).exists():
        burned = Path(output_path).with_name(Path(output_path).stem + "_subtitled.mp4")
        ok2, report2 = burn_subtitles(output_path, srt_path, burned)
        reports.append(report2)
        if ok2:
            final_path = str(burned)
    return final_path, "\n\n".join(reports)
