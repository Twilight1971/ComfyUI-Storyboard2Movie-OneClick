from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import DEFAULT_NEGATIVE_PROMPT


MODE_STYLE = {
    "balanced": "cinematic realism, natural motion, coherent physics, detailed textures, stable identity, consistent wardrobe, believable environment",
    "motion-heavy": "cinematic realism, clear physical motion, visible body movement, coherent physics, stable identity, no scene change",
    "ugc-realistic": "realistic handheld phone footage, natural autofocus, subtle motion blur, authentic pacing, stable identity, believable environment",
    "cinematic": "cinematic realism, natural motion, coherent physics, detailed textures, stable identity, consistent wardrobe, believable environment",
    "anime": "high quality anime film style, clean linework, expressive motion, consistent character design, coherent shot continuity",
    "pixel-art": "16-bit pixel art, sprite-like motion, clean retro game composition, no photorealism, stable character design",
}


SHOT_STARTS = {
    1: "standing in the street in the same pose as the storyboard frame",
    2: "holding the same facial direction and expression as the storyboard close-up",
    3: "holding the glowing hero prop near the body as shown in the storyboard frame",
    4: "already in the action pose shown in the storyboard frame",
    5: "facing the threat from the same over-shoulder composition",
    6: "holding the final hero stance from the storyboard frame",
}

SHOT_ENDS = {
    1: "the character still walking forward with the same face and outfit clearly readable",
    2: "the character finishing the subtle reaction without changing identity",
    3: "the hero prop glowing slightly brighter while the character remains stable",
    4: "the action pose held clearly without the body or face distorting",
    5: "the confrontation held in the same composition with stable wardrobe and face",
    6: "the final hero pose held steadily with the prop and silhouette unchanged",
}


def _clean(text: Any, fallback: str) -> str:
    value = str(text or "").strip()
    return value if value else fallback


def _sentence(text: str) -> str:
    text = " ".join(str(text or "").split()).strip(" .")
    return f"{text}." if text else ""


def _truncate_at_sentence(text: str, limit: int = 1100) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
    if sentence_end > limit * 0.72:
        return cut[: sentence_end + 1]
    return cut.rsplit(" ", 1)[0].rstrip(" ,;:") + "."


def _audio_line(scene: Dict[str, Any]) -> str:
    audio = scene.get("audio") or {}
    dialogue = _clean(audio.get("dialogue"), "")
    voiceover = _clean(audio.get("voiceover"), "")
    sfx = audio.get("sfx") or []
    music = _clean(audio.get("music_mood"), "")
    if dialogue:
        return f'Audio: spoken dialogue "{dialogue}", with realistic ambient sound.'
    if voiceover:
        return f'Audio: voiceover "{voiceover}", with realistic ambient sound.'
    cues = ", ".join(str(x) for x in sfx if str(x).strip())
    if cues:
        return f"Audio: {cues}, subtle {music or 'ambient room tone'}, no dialogue."
    return "Audio: realistic ambient sound, subtle footsteps or environmental movement, no dialogue."


def _subject_line(scene: Dict[str, Any], project: Dict[str, Any]) -> str:
    wardrobe = _clean(project.get("wardrobe_notes"), "the exact wardrobe shown in the first frame")
    prop = _clean(project.get("hero_prop"), "the same hero prop if visible")
    environment = _clean(scene.get("environment_notes"), "the same environment shown in the storyboard frame")
    shot_title = _clean(scene.get("shot_title"), f"Shot {scene.get('id', 1):02d}")
    shot_type = _clean(scene.get("shot_type"), "cinematic")
    return (
        f"The exact same main character from {shot_title} stands in {environment}, "
        f"wearing {wardrobe}, holding or interacting with {prop}. "
        f"The framing is a {shot_type} shot based on the provided first frame."
    )


def build_scene_prompt(scene: Dict[str, Any], project: Dict[str, Any], global_style: str, keep_character_consistency: bool, add_camera_language: bool, mode: str) -> str:
    duration = float(scene.get("duration", 3.0))
    scene_id = int(scene.get("id", 1))
    subject = _subject_line(scene, project)
    action = _clean(scene.get("motion_description"), "the character makes one small, physically realistic movement")
    start_pose = _clean(scene.get("start_action"), SHOT_STARTS.get(scene_id, "matching the exact pose and composition of the first frame"))
    end_pose = _clean(scene.get("end_action"), SHOT_ENDS.get(scene_id, "the character holds a readable final pose without identity change"))
    camera = _clean(scene.get("camera"), "locked-off frame with a very slow push-in") if add_camera_language else "locked-off frame"
    environment = _clean(scene.get("environment_notes"), "environment inferred from the storyboard")
    lighting = _clean(scene.get("lighting"), "realistic cinematic lighting")
    style_parts = [project.get("style"), MODE_STYLE.get(mode, MODE_STYLE["cinematic"])]
    if mode not in {"cinematic", "balanced"}:
        style_parts.insert(1, global_style)
    style = ", ".join(part for part in style_parts if part)
    identity_lock = _clean(
        project.get("character_identity_lock"),
        "same character identity, same wardrobe, same face, same hero prop",
    )
    continuity = "Identity: preserve the exact same face, age, skin texture, hair or bald head shape, facial hair, body build, wardrobe, colors, and hero prop from the character reference." if keep_character_consistency else "Keep object positions and scene geography consistent."
    parts = [
        f"Scene {scene_id}, {duration:g} seconds.",
        _sentence(subject),
        f"The scene begins with {start_pose}.",
        f"Over the next few seconds, {action}, while the background stays in the same location and only subtle environmental motion occurs.",
        f"The action ends with {end_pose}.",
        f"Camera: {camera}, keeping the subject clearly framed." if add_camera_language else "Camera: locked-off frame, keeping the subject clearly framed.",
        continuity,
        f"Lighting: {lighting}.",
        f"Style: {style}.",
        _audio_line(scene),
        "For image-to-video, preserve the first frame composition; animate only the described motion. Keep the scene coherent, physically realistic, and consistent throughout.",
        f"Identity lock: {identity_lock}" if keep_character_consistency and len(str(identity_lock)) < 220 else "",
    ]
    prompt = " ".join(p.strip() for p in parts if p and p.strip())
    return _truncate_at_sentence(prompt, 1100)


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
