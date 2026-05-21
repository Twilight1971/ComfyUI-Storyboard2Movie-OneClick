from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import ensure_dir, load_user_ltx_mapping, safe_json_dumps


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


def export_scene_workflows(plan: Dict[str, Any], output_dir: str | Path, seed: int, width: int, height: int, fps: int) -> Tuple[List[str], str]:
    output = ensure_dir(output_dir)
    mapping = load_user_ltx_mapping()
    project = plan.get("project", {})
    paths: List[str] = []
    for scene in plan.get("scenes", []):
        workflow = build_placeholder_ltx_workflow(scene, project, seed, width, height, fps)
        workflow["_storyboard2movie"] = {"ltx_mapping": mapping, "scene": scene}
        path = output / f"scene_{int(scene.get('id', len(paths) + 1)):03d}_ltx23.json"
        path.write_text(safe_json_dumps(workflow), encoding="utf-8")
        paths.append(str(path))
    return paths, f"Exported {len(paths)} editable LTX-2.3 scene workflow JSON files to {output}."


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
        # Placeholder workflows are not executable until mapped to installed LTX nodes.
        if "S2M_LTX23_Scene_Metadata" in json.dumps(workflow):
            reports.append(f"{path}: skipped API submit because it is a template/metadata workflow.")
            continue
        ok, report = submit_workflow_to_comfyui(workflow, server_url)
        reports.append(f"{path}: submitted={ok} {report}")
        time.sleep(0.25)
    return "\n".join(reports)
