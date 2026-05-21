from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import LTX_I2V_TEMPLATE, MODELS_ROOT, ensure_dir, load_user_ltx_mapping, safe_json_dumps


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


def _find_tokenizer_root() -> Optional[Path]:
    roots = [
        Path("N:/pinokio/api/comfy.git/app/models"),
        MODELS_ROOT,
    ]
    for root in roots:
        if not root.exists():
            continue
        matches = list(root.rglob("tokenizer.model"))
        if matches:
            return matches[0].parent
    return None


def _disconnect_input(node: Dict[str, Any], input_name: str) -> None:
    for item in node.get("inputs", []):
        if item.get("name") == input_name:
            item["link"] = None


def _node_by_id(workflow: Dict[str, Any], node_id: int) -> Optional[Dict[str, Any]]:
    for node in workflow.get("nodes", []):
        if int(node.get("id", -1)) == int(node_id):
            return node
    return None


def _remove_links(workflow: Dict[str, Any], link_ids: List[int]) -> None:
    remove = set(int(x) for x in link_ids)
    workflow["links"] = [link for link in workflow.get("links", []) if int(link[0]) not in remove]
    for node in workflow.get("nodes", []):
        for inp in node.get("inputs", []):
            if inp.get("link") in remove:
                inp["link"] = None
        for out in node.get("outputs", []):
            links = out.get("links")
            if isinstance(links, list):
                out["links"] = [x for x in links if x not in remove]


def _remove_nodes(workflow: Dict[str, Any], node_ids: List[int]) -> None:
    remove = set(int(x) for x in node_ids)
    workflow["nodes"] = [node for node in workflow.get("nodes", []) if int(node.get("id", -1)) not in remove]


def _configure_direct_checkpoint_clip(workflow: Dict[str, Any], prompt: str, reason: str = "checkpoint_clip") -> None:
    """Use the checkpoint CLIP path instead of the Gemma 12B text encoder.

    This bypasses the Gemma prompt enhancer/loader and connects the CLIP output
    from CheckpointLoaderSimple to CLIPTextEncode directly. It is the default
    path for 16GB GPUs, where Gemma 12B can OOM before video sampling starts.
    """
    # Remove Gemma loader/enhancer links:
    # 13924: Gemma CLIP -> CLIPTextEncode, 13925: enhancer -> CLIPTextEncode text,
    # 13926/13927/13928: inputs to enhancer.
    _remove_links(workflow, [13924, 13925, 13926, 13927, 13928])
    clip_encode = _node_by_id(workflow, 5174)
    checkpoint = _node_by_id(workflow, 5176)
    if clip_encode:
        _disconnect_input(clip_encode, "text")
        for item in clip_encode.get("inputs", []):
            if item.get("name") == "clip":
                item["link"] = 200001
        clip_encode["widgets_values"] = [prompt]
    if checkpoint:
        outputs = checkpoint.get("outputs", [])
        if len(outputs) > 1:
            links = outputs[1].get("links")
            outputs[1]["links"] = (links if isinstance(links, list) else []) + [200001]
    if not any(int(link[0]) == 200001 for link in workflow.get("links", [])):
        workflow.setdefault("links", []).append([200001, 5176, 1, 5174, 0, "CLIP"])
    workflow.setdefault("extra", {})["storyboard2movie_text_encoder_mode"] = reason


def _configure_native_ltxv_cpu_clip(workflow: Dict[str, Any], prompt: str, clip_name: str, projection_name: str) -> None:
    """Use ComfyUI's native LTXV DualCLIPLoader on CPU.

    The LTX-2.3 checkpoint does not contain a CLIP/text encoder. The dedicated
    LTXVideo Gemma loader can exceed 16GB VRAM, so this route uses ComfyUI's
    CLIPType.LTXV loader with the fp4 mixed Gemma file plus text projection and
    keeps it on CPU.
    """
    _remove_links(workflow, [13924, 13925, 13926, 13927, 13928])
    _remove_nodes(workflow, [5178, 5192])
    clip_encode = _node_by_id(workflow, 5174)
    if clip_encode:
        _disconnect_input(clip_encode, "text")
        for item in clip_encode.get("inputs", []):
            if item.get("name") == "clip":
                item["link"] = 200002
        clip_encode["widgets_values"] = [prompt]

    workflow.setdefault("nodes", []).append(
        {
            "id": 900002,
            "type": "DualCLIPLoader",
            "pos": [690, 160],
            "size": [360, 106],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
                {
                    "name": "CLIP",
                    "type": "CLIP",
                    "links": [200002],
                    "slot_index": 0,
                }
            ],
            "properties": {"Node name for S&R": "DualCLIPLoader"},
            "widgets_values": [clip_name, projection_name, "ltxv", "cpu"],
        }
    )
    if not any(int(link[0]) == 200002 for link in workflow.get("links", [])):
        workflow.setdefault("links", []).append([200002, 900002, 0, 5174, 0, "CLIP"])
    workflow.setdefault("extra", {})["storyboard2movie_text_encoder_mode"] = "native_ltxv_cpu"


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
    _patch_node(workflow, 5174, [prompt])
    _patch_node(workflow, 5184, [int(fps)])
    _patch_node(workflow, 5186, [frames, "fixed"])
    _patch_node(workflow, 5185, [int(width), int(height), 1, 0])
    _patch_node(workflow, 5189, [frames, int(fps), 0.72, int(seed) + scene_id - 1])
    _patch_node(workflow, 4958, [output_prefix, "auto", "auto"])

    checkpoint = mapping.get("checkpoint")
    audio_vae = mapping.get("audio_vae", checkpoint)
    text_encoder_mode = str(mapping.get("text_encoder_mode", "native_ltxv_cpu")).lower().strip()
    native_ltxv_clip = mapping.get("native_ltxv_clip", "gemma_3_12B_it_fp4_mixed.safetensors")
    native_ltxv_projection = mapping.get("native_ltxv_projection", "ltx2\\ltx-2.3_text_projection_bf16.safetensors")
    gemma_text_encoder = mapping.get("gemma_text_encoder")
    gemma_connector = mapping.get("gemma_connector") or checkpoint
    gemma_max_length = int(mapping.get("gemma_max_length", 512))
    latent_upscaler = mapping.get("latent_upscaler")
    if checkpoint:
        _patch_first_widget(workflow, 5176, checkpoint)
    if audio_vae:
        _patch_first_widget(workflow, 5188, audio_vae)
    tokenizer_root = _find_tokenizer_root()
    quality_mode = str(project.get("quality_mode", "4060ti_safe")).lower()
    use_gemma = text_encoder_mode in {"gemma", "gemma_12b", "gemma12b"}
    use_native_ltxv = text_encoder_mode in {"native_ltxv_cpu", "ltxv_cpu", "dualclip_ltxv_cpu"}
    if use_native_ltxv:
        _configure_native_ltxv_cpu_clip(workflow, prompt, native_ltxv_clip, native_ltxv_projection)
    elif use_gemma and tokenizer_root and (gemma_text_encoder or gemma_connector):
        _patch_node(workflow, 5178, [gemma_text_encoder or "", gemma_connector or "", gemma_max_length])
        workflow.setdefault("extra", {})["storyboard2movie_text_encoder_mode"] = "gemma"
    elif use_gemma and not tokenizer_root:
        _configure_direct_checkpoint_clip(workflow, prompt, "checkpoint_clip_fallback_no_gemma_tokenizer")
    else:
        reason = "checkpoint_clip_4060ti_safe" if quality_mode == "4060ti_safe" else "checkpoint_clip"
        _configure_direct_checkpoint_clip(workflow, prompt, reason)
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
