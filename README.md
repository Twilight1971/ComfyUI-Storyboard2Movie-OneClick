# ComfyUI-Storyboard2Movie-OneClick

One-click storyboard-to-movie helpers for ComfyUI. Load a single storyboard image, parse it into a reusable scene plan, build LTX-2.3-ready prompts, export one workflow JSON per scene, create placeholder audio/subtitles, and assemble rendered scene clips into a final MP4 with FFmpeg.

This package does not reimplement LTX-2.3 Video. It integrates around existing ComfyUI LTXVideo nodes/templates and keeps the mapping layer editable because LTX node class names and widget schemas can change between installs.

## What It Does

- Analyzes one storyboard image using OCR when available, with layout heuristics when OCR/VLM tools are missing.
- Produces a JSON scene plan with durations, prompts, camera notes, audio cues, captions, and transitions.
- Enhances each scene into concise LTX-2.3 video prompts.
- Exports per-scene workflow JSON files under `outputs/storyboard_movie/<name>/workflows/`.
- Creates silent or placeholder FFmpeg audio and SRT captions.
- Assembles rendered scene MP4s into a final H.264/AAC MP4.

## Installation

1. Copy this folder to your ComfyUI custom nodes directory:

   ```powershell
   Copy-Item -Recurse ComfyUI-Storyboard2Movie-OneClick N:\KI_Daten\custom_nodes\
   ```

2. Install the minimal requirements in the same Python environment ComfyUI uses:

   ```powershell
   python -m pip install -r N:\KI_Daten\custom_nodes\ComfyUI-Storyboard2Movie-OneClick\requirements.txt
   ```

3. Install FFmpeg and make sure `ffmpeg.exe` is on `PATH`.

4. Install or update ComfyUI and your LTXVideo custom nodes/templates. Put models under your model root, for example:

   ```text
   N:\KI_Daten\models
   N:\KI_Daten\custom_nodes
   ```

5. Restart ComfyUI. The following nodes should appear under `Storyboard2Movie`:

- `Storyboard Image Analyzer`
- `Storyboard Scene Prompt Builder`
- `LTX Storyboard Movie Orchestrator`
- `Storyboard Audio Builder`
- `Storyboard Movie Assembler`

## Required Models

- Required for actual video generation: LTX-2.3 Video models through your installed ComfyUI LTXVideo package or ComfyUI Template Library workflow.
- Optional OCR: `pytesseract` plus the Tesseract executable, or `easyocr`.
- Optional local VLM: Qwen2.5-VL, Florence-2, or Moondream adapter. The MVP includes an adapter abstraction note and safe fallback, not a hard dependency.
- Optional local TTS/audio: Piper, Coqui, or another local engine. The default uses FFmpeg placeholder or silent audio.

No paid API or cloud service is required.

## RTX 4060 Ti 16GB Notes

Use `quality_mode = 4060ti_safe` first:

- 9:16: `576x1024`
- 16:9: `768x432`
- 1:1: `640x640`
- `24 fps`
- Prefer `2-4` seconds per scene.
- Split a 12-second storyboard into multiple short clips instead of one long generation.
- Enable fp16/bf16/model offload in your LTXVideo template where supported.
- Fixed 4:5 editorial storyboard sheets use `640x800` in `4060ti_safe`.
- Generated scene workflows use `text_encoder_mode = native_ltxv_cpu` by default. This uses ComfyUI's native LTXV `DualCLIPLoader` on CPU with the fp4 mixed Gemma file and LTX text projection, avoiding the full Gemma loader's 16GB VRAM OOM.

## Recommended Storyboard Format

For best results, use one fixed 4:5 vertical editorial character-sheet storyboard:

- Large character/profile panel on the left.
- Storyboard grid on the right with 6 shots in two columns and three rows.
- Each shot has a black header bar and a frame image. Shot titles may change per project; the parser keeps generic `Shot 01`, `Shot 02`, etc. when OCR cannot read the titles.
- Footer strip contains wardrobe/gear, personality/skills, and locations/conditions.

This layout is detected as `editorial_character_sheet_4x5`. The parser extracts the right-side shot grid, assigns 6 scenes, uses 2 seconds per scene for a 12-second board, and keeps the overall movie target at 4:5.

Character consistency is treated as a first-class requirement. The left profile panel becomes the identity reference, and every scene prompt includes an identity lock for the same face, age, body build, wardrobe, colors, hero prop, and silhouette.

If CUDA runs out of memory, reduce resolution, reduce frames/duration, close other GPU applications, and render one scene at a time.

## One-Click Usage

1. Load your storyboard image in ComfyUI.
2. Connect it to `Storyboard Image Analyzer`.
3. Connect the analyzer JSON to `Storyboard Scene Prompt Builder`.
4. Connect the enhanced JSON to `LTX Storyboard Movie Orchestrator`.
5. Set `quality_mode = 4060ti_safe`.
6. Press Queue.

You can also import `workflows/storyboard2movie_oneclick_ltx23.json`. This graph is pre-wired:

```text
Load Image -> Storyboard Image Analyzer -> Storyboard Scene Prompt Builder -> LTX Storyboard Movie Orchestrator
                                                `-> Storyboard Audio Builder -> Storyboard Movie Assembler
```

The first run is expected to create the plan, prompt list, audio/SRT placeholders, and per-scene LTX workflow JSON files. The final MP4 can only be assembled after the generated LTX scene workflows have been rendered into the expected `scene_001.mp4`, `scene_002.mp4`, etc. files.

For the installed Windows/NVIDIA setup used by this package, scene workflows are patched from:

```text
N:\KI_Daten\custom_nodes\ComfyUI-LTXVideo\example_workflows\LTX-2_I2V_Distilled_wLora.json
```

The orchestrator also crops every storyboard shot into a real image-to-video start frame and writes it both to the project output folder and to ComfyUI input:

```text
output\storyboard_movie\<output_name>\frames\scene_001_start.png
input\storyboard2movie\<output_name>\scene_001_start.png
```

Each generated LTX scene workflow uses its matching start frame through `LoadImage`.

The orchestrator writes:

```text
outputs/storyboard_movie/<output_name>/storyboard_plan_final.json
outputs/storyboard_movie/<output_name>/workflows/scene_001_ltx23.json
outputs/storyboard_movie/<output_name>/audio/audio_mix.m4a
outputs/storyboard_movie/<output_name>/audio/captions.srt
outputs/storyboard_movie/<output_name>/final/render_report.txt
```

Render each generated scene workflow with your installed LTX-2.3 template and save clips as:

```text
outputs/storyboard_movie/<output_name>/scenes/scene_001.mp4
outputs/storyboard_movie/<output_name>/scenes/scene_002.mp4
```

Then rerun the orchestrator or use `Storyboard Movie Assembler` to create:

```text
outputs/storyboard_movie/<output_name>/final/<output_name>_final.mp4
```

## LTX Workflow Mapping

Generated scene workflows are patched from the installed ComfyUI-LTXVideo I2V distilled template when it is found. If the template is missing, the package falls back to metadata-only workflows.

For advanced automation, create `ltx_node_mapping.json` next to `config.py` or point `STORYBOARD2MOVIE_LTX_MAPPING` to a mapping file. The code is structured so this can be replaced with a concrete template generator for your exact LTXVideo installation.

Default 16GB-safe mapping:

```json
{
  "text_encoder_mode": "native_ltxv_cpu",
  "native_ltxv_clip": "gemma_3_12B_it_fp4_mixed.safetensors",
  "native_ltxv_projection": "ltx2\\ltx-2.3_text_projection_bf16.safetensors"
}
```

To force the full Gemma loader on a larger GPU, set:

```json
{
  "text_encoder_mode": "gemma",
  "gemma_text_encoder": "gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors",
  "gemma_connector": "LTX23\\ltx-2.3-22b-dev-fp8.safetensors",
  "gemma_max_length": 512
}
```

## Troubleshooting

- Missing FFmpeg: install FFmpeg and restart ComfyUI. Audio and assembly need it.
- Missing OCR: install `pytesseract` or `easyocr`; otherwise the analyzer uses image layout heuristics.
- Missing LTXVideo nodes: the package still creates plans and scene workflow JSON files, but you must install LTXVideo to generate actual clips.
- CUDA OOM: use `4060ti_safe`, shorten scene durations, lower resolution, render one scene at a time, and keep `text_encoder_mode = native_ltxv_cpu` on 16GB GPUs.
- Bad storyboard parsing: edit `storyboard_plan_final.json` manually, then feed it into `Storyboard Scene Prompt Builder`.
- Empty final video path: expected scene clips do not exist yet. Render them first, then assemble.

## Example JSON

See `examples/example_storyboard_plan.json` and `examples/example_settings.json`.

## Development Notes

The implementation is modular:

- `storyboard_parser.py`: OCR, panel/layout heuristic, scene plan creation.
- `prompt_builder.py`: LTX prompt enhancement modes.
- `workflow_builder.py`: per-scene LTX workflow export and optional ComfyUI API submission.
- `audio_builder.py`: silent/placeholder audio and SRT generation.
- `video_assembler.py` and `ffmpeg_utils.py`: normalization, concatenation, audio mux, subtitle burn-in.
- `nodes.py`: ComfyUI node registration.

The MVP is intentionally useful before full LTX automation: it turns one storyboard image into a production scene plan, LTX prompts, workflow files, audio/captions, and final assembly tooling without pretending to render video when your local LTX node schema is unknown.
