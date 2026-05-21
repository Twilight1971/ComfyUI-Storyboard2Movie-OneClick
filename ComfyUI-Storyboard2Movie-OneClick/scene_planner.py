from __future__ import annotations

from typing import Any, Dict


def clamp_scene_durations(plan: Dict[str, Any], max_scene_seconds: float) -> Dict[str, Any]:
    """Split overly long scenes into shorter render-friendly segments."""
    scenes = []
    next_id = 1
    start = 0.0
    for scene in plan.get("scenes", []):
        duration = float(scene.get("duration", max_scene_seconds))
        if duration <= max_scene_seconds:
            new_scene = dict(scene)
            new_scene["id"] = next_id
            new_scene["start"] = round(start, 3)
            new_scene["duration"] = round(duration, 3)
            new_scene["end"] = round(start + duration, 3)
            scenes.append(new_scene)
            next_id += 1
            start += duration
            continue
        parts = max(1, int((duration + max_scene_seconds - 0.001) // max_scene_seconds))
        part_duration = duration / parts
        for part in range(parts):
            new_scene = dict(scene)
            new_scene["id"] = next_id
            new_scene["start"] = round(start, 3)
            new_scene["duration"] = round(part_duration, 3)
            new_scene["end"] = round(start + part_duration, 3)
            new_scene["visual_description"] = f"{scene.get('visual_description', '')} continuation part {part + 1}"
            scenes.append(new_scene)
            next_id += 1
            start += part_duration
    plan["scenes"] = scenes
    plan.setdefault("project", {})["total_duration"] = round(start, 3)
    return plan
