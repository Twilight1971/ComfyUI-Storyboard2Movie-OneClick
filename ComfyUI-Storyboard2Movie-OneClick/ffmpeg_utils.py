from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import ensure_dir


def _run_ffmpeg(args: List[str]) -> Tuple[bool, str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False, "FFmpeg was not found on PATH."
    cmd = [exe, "-hide_banner", "-y", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as exc:
        return False, f"FFmpeg execution failed: {exc}"
    report = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    return proc.returncode == 0, report


def check_ffmpeg_available() -> Tuple[bool, str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False, "FFmpeg was not found on PATH. Install FFmpeg and restart ComfyUI."
    proc = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False)
    first = (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr) else exe
    return proc.returncode == 0, first


def normalize_video(input_path: str | Path, output_path: str | Path, width: int, height: int, fps: int) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps},format=yuv420p"
    return _run_ffmpeg(["-i", str(input_path), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(output_path)])


def concat_videos(video_paths: Iterable[str | Path], output_path: str | Path) -> Tuple[bool, str]:
    paths = [Path(p) for p in video_paths]
    ensure_dir(Path(output_path).parent)
    list_file = Path(output_path).with_suffix(".concat.txt")
    list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    return _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_file), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(output_path)])


def add_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    return _run_ffmpeg([
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ])


def create_silent_audio(duration: float, output_path: str | Path) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    duration = max(0.1, float(duration))
    return _run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{duration:.3f}", "-c:a", "aac", str(output_path)])


def create_tone_audio(duration: float, output_path: str | Path, frequency: int = 220) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    return _run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000", "-t", f"{max(duration, 0.1):.3f}", "-af", "volume=0.06", "-c:a", "aac", str(output_path)])


def _srt_time(seconds: float) -> str:
    ms_total = int(round(max(0.0, seconds) * 1000))
    ms = ms_total % 1000
    total = ms_total // 1000
    s = total % 60
    total //= 60
    m = total % 60
    h = total // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def create_srt_from_plan(plan: Dict[str, Any], output_path: str | Path) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    entries: List[str] = []
    idx = 1
    for scene in plan.get("scenes", []):
        audio = scene.get("audio") or {}
        text = (audio.get("voiceover") or audio.get("dialogue") or "").strip()
        if not text:
            continue
        entries.append(f"{idx}\n{_srt_time(scene.get('start', 0))} --> {_srt_time(scene.get('end', 0))}\n{text}\n")
        idx += 1
    Path(output_path).write_text("\n".join(entries), encoding="utf-8")
    return True, f"Wrote {max(0, idx - 1)} subtitle entries to {output_path}."


def burn_subtitles(video_path: str | Path, srt_path: str | Path, output_path: str | Path) -> Tuple[bool, str]:
    ensure_dir(Path(output_path).parent)
    escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    return _run_ffmpeg(["-i", str(video_path), "-vf", f"subtitles='{escaped}'", "-c:v", "libx264", "-c:a", "copy", str(output_path)])


def final_mux(video_paths: List[str | Path], audio_path: Optional[str | Path], output_path: str | Path, settings: Dict[str, Any]) -> Tuple[bool, str]:
    temp_dir = ensure_dir(Path(output_path).parent / "_normalized")
    reports: List[str] = []
    normalized: List[Path] = []
    width = int(settings.get("width", 576))
    height = int(settings.get("height", 1024))
    fps = int(settings.get("fps", 24))
    for i, path in enumerate(video_paths, start=1):
        out = temp_dir / f"clip_{i:03d}.mp4"
        ok, report = normalize_video(path, out, width, height, fps)
        reports.append(f"normalize {path}: {ok}\n{report}")
        if not ok:
            return False, "\n\n".join(reports)
        normalized.append(out)
    stitched = Path(output_path).with_name(Path(output_path).stem + "_video_only.mp4")
    ok, report = concat_videos(normalized, stitched)
    reports.append(f"concat: {ok}\n{report}")
    if not ok:
        return False, "\n\n".join(reports)
    if audio_path and Path(audio_path).exists():
        ok, report = add_audio(stitched, audio_path, output_path)
        reports.append(f"audio mux: {ok}\n{report}")
        return ok, "\n\n".join(reports)
    Path(output_path).write_bytes(stitched.read_bytes())
    reports.append("No audio path supplied; copied video-only MP4.")
    return True, "\n\n".join(reports)


def write_scene_paths_json(paths: List[str | Path], output_path: str | Path) -> str:
    ensure_dir(Path(output_path).parent)
    payload = {"scene_video_paths": [str(Path(p)) for p in paths]}
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)
