from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import DEFAULT_NEGATIVE_PROMPT


MODE_STYLE = {
    "balanced": "balanced cinematic realism with clear subject motion",
    "motion-heavy": "dynamic action, visible motion arcs, strong temporal continuity",
    "ugc-realistic": "handheld iPhone camera, realistic autofocus, subtle motion blur, natural imperfections, authentic social media pacing, vertical 9:16",
    "cinematic": "controlled camera, cinematic lighting, expressive lensing, dramatic but coherent motion",
    "anime": "high quality anime film style, clean linework, expressive motion, consistent character design",
    "pixel-art": "16-bit pixel art, sprite-like motion, clean retro game composition, no photorealism",
}


def _clean(text: Any, fallback: str) -> str:
    value = str(text or "").strip()
    return value if value else fallback


def build_scene_prompt(scene: Dict[str, Any], project: Dict[str, Any], global_style: str, keep_character_consistency: bool, add_camera_language: bool, mode: str) -> str:
    duration = float(scene.get("duration", 3.0))
    subject = _clean(scene.get("visual_description"), "main subject from the storyboard panel")
    action = _clean(scene.get("motion_description"), "natural continuous action")
    camera = _clean(scene.get("camera"), "stable cinematic camera movement") if add_camera_language else ""
    environment = _clean(scene.get("environment_notes"), "environment inferred from the storyboard")
    lighting = _clean(scene.get("lighting"), "realistic cinematic lighting")
    style = ", ".join(part for part in [project.get("style"), global_style, MODE_STYLE.get(mode, MODE_STYLE["cinematic"])] if part)
    identity_lock = _clean(
        project.get("character_identity_lock"),
        "same character identity, same wardrobe, same face, same hero prop",
    )
    continuity = _clean(scene.get("character_continuity"), "stable identity and consistent wardrobe") if keep_character_consistency else "consistent objects and scene geography"
    if identity_lock and continuity == identity_lock:
        continuity = ""
    parts = [
        f"Scene {scene.get('id', 1)}, {duration:g} seconds.",
        "Use the provided storyboard frame as the exact first frame and visual anchor.",
        subject,
        action,
        camera,
        environment,
        lighting,
        "Preserve the original composition, face, wardrobe, prop, colors, and background from the first frame.",
        style,
        identity_lock,
        continuity,
        "Subtle natural motion, stable face, stable identity, coherent anatomy, no redesign, no scene change.",
    ]
    prompt = " ".join(p.strip() for p in parts if p and p.strip())
    return " ".join(prompt.split())[:900]


def enhance_storyboard_plan(plan: Dict[str, Any], global_style: str, prompt_strength: float, keep_character_consistency: bool, add_camera_language: bool, ltx_prompt_mode: str) -> Tuple[Dict[str, Any], str]:
    project = plan.setdefault("project", {})
    project.setdefault("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    project["prompt_strength"] = float(prompt_strength)
    project["ltx_prompt_mode"] = ltx_prompt_mode
    prompts: List[str] = []
    for scene in plan.get("scenes", []):
        scene["prompt"] = build_scene_prompt(scene, project, global_style, keep_character_consistency, add_camera_language, ltx_prompt_mode)
        scene["negative_prompt"] = scene.get("negative_prompt") or project.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT
        prompts.append(f"{scene.get('id', len(prompts) + 1):03d}: {scene['prompt']}")
    return plan, "\n\n".join(prompts)
