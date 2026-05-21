from __future__ import annotations

import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import LTX_I2V_TEMPLATE, comfy_input_dir, ensure_dir, load_user_ltx_mapping, safe_json_dumps


def build_placeholder_ltx_workflow(scene: Dict[str, Any], project: Dict[str, Any], seed: int, width: int, height: int, fps: int) -> Dict[str, Any]:
    """Export an editable API-format workflow with stable metadata.

    LTXVideo ComfyUI node names vary by package and version. This placeholder is
    intentionally small and carries all scene parameters in node widgets and
    _meta so users can paste them into their installed LTX-2.3 template.
    """
    scene_id = int(scene.get("id", 1))
    frames = max(8, int(round(float(scene.get("duration", 3.0)) * fps)))
    return {
        "1": {
            "class_type": "S2M_LTX23_Scene_Metadata",
            "inputs": {
                "prompt": scene.get("prompt", ""),
                "negative_prompt": scene.get("negative_prompt", project.get("negative_prompt", "")),
                "seed": int(seed) + scene_id - 1,
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "frames": frames,
                "duration": float(scene.get("duration", 3.0)),
                "output_filename": f"scene_{scene_id:03d}.mp4",
            },
            "_meta": {
                "title": f"Storyboard2Movie LTX-2.3 Scene {scene_id:03d}",
                "instructions": "Replace this metadata node with your installed LTXVideo nodes or use it as a parameter carrier for a template.",
            },
        }
    }


def _valid_ltx_frames(duration: float, fps: int) -> int:
    raw = max(9, int(round(float(duration) * int(fps))))
    # LTX workflows expect frame count divisible by 8 plus 1.
    return max(9, ((raw - 1 + 7) // 8) * 8 + 1)


def _patch_node(workflow: Dict[str, Any], node_id: int, values: List[Any]) -> None:
    for node in workflow.get("nodes", []):
        if int(node.get("id", -1)) == int(node_id):
            node["widgets_values"] = values
            return


def _patch_first_widget(workflow: Dict[str, Any], node_id: int, value: Any) -> None:
    for node in workflow.get("nodes", []):
        if int(node.get("id", -1)) == int(node_id):
            widgets = list(node.get("widgets_values") or [])
            if widgets:
                widgets[0] = value
            else:
                widgets = [value]
            node["widgets_values"] = widgets
            return


def build_ltx_i2v_template_workflow(scene: Dict[str, Any], project: Dict[str, Any], seed: int, width: int, height: int, fps: int) -> Dict[str, Any]:
    if not LTX_I2V_TEMPLATE.exists():
        return build_placeholder_ltx_workflow(scene, project, seed, width, height, fps)

    workflow = json.loads(LTX_I2V_TEMPLATE.read_text(encoding="utf-8"))
    mapping = load_user_ltx_mapping()
    scene_id = int(scene.get("id", 1))
    frames = _valid_ltx_frames(float(scene.get("duration", 3.0)), int(fps))
    prompt = scene.get("prompt", "")
    input_name = scene.get("start_frame_input_name") or scene.get("start_frame_path") or "storyboard.png"
    output_prefix = f"storyboard_movie/{project.get('output_name', 'movieclip')}/scenes/scene_{scene_id:03d}"

    # Node IDs from ComfyUI-LTXVideo/example_workflows/LTX-2_I2V_Distilled_wLora.json.
    _patch_node(workflow, 5180, [input_name, "image"])
    _patch_node(workflow, 5175, [prompt])
    _patch_node(workflow, 5184, [int(fps)])
    _patch_node(workflow, 5186, [frames, "fixed"])
    _patch_node(workflow, 5185, [int(width), int(height), 1, 0])
    _patch_node(workflow, 5189, [frames, int(fps), 0.72, int(seed) + scene_id - 1])
    _patch_node(workflow, 4958, [output_prefix, "auto", "auto"])

    checkpoint = mapping.get("checkpoint")
    audio_vae = mapping.get("audio_vae", checkpoint)
    gemma_text_encoder = mapping.get("gemma_text_encoder")
    gemma_connector = mapping.get("gemma_connector") or checkpoint
    latent_upscaler = mapping.get("latent_upscaler")
    if checkpoint:
        _patch_first_widget(workflow, 5176, checkpoint)
    if audio_vae:
        _patch_first_widget(workflow, 5188, audio_vae)
    if gemma_text_encoder or gemma_connector:
        _patch_node(workflow, 5178, [gemma_text_encoder or "", gemma_connector or "", 1024])
    if latent_upscaler:
        _patch_first_widget(workflow, 5210, latent_upscaler)

    workflow.setdefault("extra", {})["storyboard2movie"] = {
        "scene_id": scene_id,
        "duration": scene.get("duration"),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": input_name,
        "expected_output_prefix": output_prefix,
    }
    return workflow


def export_scene_workflows(plan: Dict[str, Any], output_dir: str | Path, seed: int, width: int, height: int, fps: int) -> Tuple[List[str], str]:
    output = ensure_dir(output_dir)
    mapping = load_user_ltx_mapping()
    project = plan.get("project", {})
    paths: List[str] = []
    for scene in plan.get("scenes", []):
        workflow = build_ltx_i2v_template_workflow(scene, project, seed, width, height, fps)
        workflow["_storyboard2movie"] = {"ltx_mapping": mapping, "scene": scene}
        path = output / f"scene_{int(scene.get('id', len(paths) + 1)):03d}_ltx23.json"
        path.write_text(safe_json_dumps(workflow), encoding="utf-8")
        paths.append(str(path))
    return paths, f"Exported {len(paths)} patched LTX-2.3 I2V scene workflow JSON files to {output}."


def submit_workflow_to_comfyui(workflow: Dict[str, Any], server_url: str = "http://127.0.0.1:666") -> Tuple[bool, str]:
    payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode("utf-8")
    req = urllib.request.Request(f"{server_url.rstrip('/')}/prompt", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return True, body
    except Exception as exc:
        return False, f"Could not submit workflow to ComfyUI server {server_url}: {exc}"


def try_submit_scene_workflows(paths: List[str], server_url: str = "http://127.0.0.1:666") -> str:
    reports: List[str] = []
    for path in paths:
        workflow = json.loads(Path(path).read_text(encoding="utf-8"))
        # UI-format workflows are ready to import in ComfyUI, but /prompt needs
        # API-format graphs. Keep API submit conservative until an API template
        # is configured.
        if "nodes" in workflow:
            reports.append(f"{path}: generated as ComfyUI UI workflow; import/render it in ComfyUI or convert to API format for automatic queueing.")
            continue
        ok, report = submit_workflow_to_comfyui(workflow, server_url)
        reports.append(f"{path}: submitted={ok} {report}")
        time.sleep(0.25)
    return "\n".join(reports)
