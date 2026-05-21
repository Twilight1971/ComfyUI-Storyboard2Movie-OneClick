from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from .config import ensure_dir
from .ffmpeg_utils import check_ffmpeg_available, create_silent_audio, create_srt_from_plan, create_tone_audio


def build_audio_timeline(plan: Dict[str, Any], output_dir: str | Path, enable_music: bool = True, enable_sfx: bool = True, enable_voiceover: bool = True, voice: str = "default", music_style: str = "cinematic electronic", audio_mode: str = "ffmpeg_placeholder") -> Tuple[str, str, str]:
    out = ensure_dir(output_dir)
    total = float(plan.get("project", {}).get("total_duration", 0.0) or sum(float(s.get("duration", 0)) for s in plan.get("scenes", [])) or 1.0)
    audio_path = out / "audio_mix.m4a"
    srt_path = out / "captions.srt"
    reports = []
    ok, ff = check_ffmpeg_available()
    reports.append(ff)
    srt_ok, srt_report = create_srt_from_plan(plan, srt_path)
    reports.append(srt_report)
    if not ok:
        return "", str(srt_path if srt_ok else ""), "Audio fallback skipped because FFmpeg is unavailable.\n" + "\n".join(reports)

    if audio_mode == "silent" or not any([enable_music, enable_sfx, enable_voiceover]):
        ok, report = create_silent_audio(total, audio_path)
        reports.append(report)
    elif audio_mode == "ffmpeg_placeholder":
        freq = 180 if "dark" in music_style.lower() else 240
        ok, report = create_tone_audio(total, audio_path, frequency=freq)
        reports.append("Created low-volume placeholder tone bed. Replace with local music/TTS assets for production audio.")
        reports.append(report)
    elif audio_mode in {"local_tts", "local_audio_model"}:
        ok, report = create_silent_audio(total, audio_path)
        reports.append(f"{audio_mode} adapter is optional and not configured; wrote silent audio instead. Voice={voice}.")
        reports.append(report)
    else:
        ok, report = create_silent_audio(total, audio_path)
        reports.append(f"Unknown audio_mode={audio_mode}; wrote silent audio.")
        reports.append(report)
    if not ok:
        reports.append("Audio generation failed; movie assembly can continue without audio.")
        return "", str(srt_path), "\n".join(reports)
    return str(audio_path), str(srt_path), "\n".join(reports)
