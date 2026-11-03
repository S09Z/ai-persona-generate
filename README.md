# idol-generate

AI Idol image generation pipeline — scrape reference photos, train a LoRA, generate images.

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Apple Silicon Mac (MPS) **or** NVIDIA GPU (CUDA, Windows or Linux)
- RTX 50-series (Blackwell) needs torch **2.7.1 + cu128**; older builds have no kernels for it
- [Playwright](https://playwright.dev/) Chromium (for Instagram scraping)

---

## 1 — Install

```zsh
# Install uv (one-time)
curl -Lsf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# Install Playwright Chromium (for scraping)
uv run playwright install chromium

# Clone kohya_ss (for LoRA training)
git clone https://github.com/kohya-ss/sd-scripts vendor/kohya_ss

# Install kohya deps in isolated venv
python3.11 -m venv vendor/kohya_ss/.venv
vendor/kohya_ss/.venv/bin/pip install -r vendor/kohya_ss/requirements.txt
vendor/kohya_ss/.venv/bin/pip install torchvision tensorboard
```

### Windows + NVIDIA (PowerShell)

Set up and tested on Windows 11 with an RTX 5070 12 GB.

```powershell
# Project deps (pyproject pins torch 2.7.1+cu128 from the cu128 index; peft is required for LoRA loading)
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect 2.7.1+cu128 True

# kohya sd-scripts in its own venv (kept separate so its pins don't fight the main env)
git clone https://github.com/kohya-ss/sd-scripts vendor/kohya_ss
python -m venv vendor/kohya_ss/.venv
vendor/kohya_ss/.venv/Scripts/python.exe -m pip install -r vendor/kohya_ss/requirements.txt
# the kohya venv needs the same cu128 torch build as the main env
```

Windows notes:
- Set `PYTHONUTF8=1` (or keep the stream reconfigure in `generate.py`) to avoid cp1252 errors on emoji output.
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences the Hugging Face symlink warning on Windows.
- Training flags that work on 12 GB: `sdpa`, `full_bf16`, `cache_latents_to_disk`, `AdamW8bit`, gradient checkpointing (see `configs/lora_auri_v2.toml`). `xformers` is not used.
- `train_lora.py` launches `vendor/kohya_ss/.venv/Scripts/python.exe` through `accelerate`.

---

## 2 — Download base model

```zsh
# Download RealVisXL_V4.0 from HuggingFace
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='SG161222/RealVisXL_V4.0',
    filename='RealVisXL_V4.0.safetensors',
    local_dir='models/checkpoints/RealVisXL_V4.0'
)
"
```

---

## 3 — Scrape reference photos

### 3a — Extract Instagram cookies (one-time login)

```zsh
uv run scrape_images.py ig --extract-cookies
# A browser opens → log in to Instagram → press Enter
# Saves: ig_cookies.json
```

### 3b — Scrape a profile

```zsh
# Scrape 30 posts (page 1)
uv run scrape_images.py ig \
  --target <username> \
  --posts 30 \
  --offset 0 \
  --out datasets/instagram/<username>

# Page 2 (posts 31–60)
uv run scrape_images.py ig \
  --target <username> \
  --posts 30 \
  --offset 30

# Scrape specific post URLs
uv run scrape_images.py ig \
  --url https://www.instagram.com/p/ABC123/ https://www.instagram.com/p/DEF456/
```

### 3c — Scrape a regular website

```zsh
uv run scrape_images.py site \
  --url https://example.com \
  --depth 2 \
  --min-size 20 \
  --workers 4
```

---

## 4 — Create a persona

Create `characters/<name>.json` — use `characters/luna.json` as a template:

```json
{
  "name": "Auri",
  "trigger_word": "auri_v1",
  "appearance": {
    "face_shape": "soft oval",
    "skin_tone": "snow white",
    "eye_color": "light sky blue",
    "hair": "short amber brown"
  },
  "lora": {
    "path": "models/loras/auri_v1.safetensors",
    "trigger_word": "auri_v1",
    "weight": 0.8
  }
}
```

---

## 5 — Prepare training dataset

```zsh
# Auto-caption images with BLIP + inject persona traits
uv run prepare_dataset.py \
  --src datasets/instagram/<username> \
  --trigger auri_v1 \
  --model blip \
  --repeats 5

# Verify output
ls datasets/lora/auri_v1/5_auri_v1/ | wc -l   # should be images × 2 (jpg + txt)
cat datasets/lora/auri_v1/5_auri_v1/0001_*.txt  # preview a caption
```

> **Recommended dataset size:** 30–200 images. More is better for face consistency.

---

## 6 — Train LoRA

```zsh
# Start training (logs to logs/lora_auri/)
uv run train_lora.py --config configs/lora_auri.toml

uv run python vendor/kohya_ss/sdxl_train_network.py --config_file configs/lora_auri.toml --max_train_steps 10

# Run in background
uv run train_lora.py --config configs/lora_auri.toml \
  > logs/lora_auri/train.log 2>&1 &

# Watch live log
tail -f logs/lora_auri/train.log
```

### Dataset hygiene and controlled experiments

`configs/lora_auri_v2.toml` is the CUDA/Windows baseline (1024px, dim 32, 12 epochs, seed 42) and reads an immutable snapshot rather than the live dataset.

```powershell
# 1. Read-only audit: near-duplicates (dhash), missing captions, writes manifest.csv
uv run scripts/audit_dataset.py datasets/lora/auri_v1/5_auri_v1 --out manifest.csv

# 2. Build a deduped, caption-cleaned snapshot with a deterministic held-out set (refuses to overwrite)
uv run scripts/build_clean_snapshot.py datasets/lora/auri_v1/5_auri_v1 `
  --out datasets/lora/auri_v1_snapshot_<date> --trigger auri_v1 --holdout 10

# 3. Scaffold a one-variable experiment (control and variant differ only by --set)
uv run scripts/new_experiment.py EXP-007-network-dropout --set lora.network_dropout=0.10
uv run train_lora.py --config experiments/EXP-007-network-dropout/control/config.toml
uv run train_lora.py --config experiments/EXP-007-network-dropout/variant/config.toml

# 4. Fixed-seed evaluation grid + contact sheets (prompt sets: experiments/eval_prompts.toml)
uv run scripts/eval_grid.py --out experiments/EXP-006-lora-weight/evaluation --set weight `
  --lora auri=models/loras/auri_v2.safetensors --weights 0.6 0.8 1.0 1.2 --base
```

Fill in `experiments/<name>/results.md` after reviewing the contact sheets. Change one variable per experiment.

### Monitor with TensorBoard

```zsh
# In a separate terminal tab
uv run tensorboard --logdir logs/lora_auri --port 6006
# Open http://localhost:6006
```

| TensorBoard tab | What to watch |
|---|---|
| **Scalars → loss/train** | Should trend downward over epochs |
| **Scalars → lr** | Cosine schedule — peaks then decays |
| **Images** | Preview samples generated every 2 epochs |

### Training config reference (`configs/lora_auri.toml`)

| Parameter | Value | Why |
|---|---|---|
| `resolution` | `768,768` | 1024px OOMs on 16 GB unified memory |
| `mixed_precision` | `"no"` | accelerate 1.6.0 blocks fp16 + bf16 on MPS |
| `network_dim` | `32` | Good face LoRA quality vs size balance |
| `max_train_epochs` | `12` | Sweet spot: 10–12 epochs avoids overfitting |
| `gradient_accumulation_steps` | `2` | Reduces peak memory vs 4 |

### Checkpoint outputs

```
models/loras/
├── auri_v1.safetensors           ← final (epoch 12)
├── auri_v1-000002.safetensors  ← saved every 2 epochs
├── auri_v1-000004.safetensors
└── ...

logs/lora_auri/samples/           ← preview images every 2 epochs
```

### When to stop training

| Epoch | Quality |
|---|---|
| 1–4 | Face not recognisable yet |
| 5–8 | Face starting to emerge |
| **9–12** | ✅ Good face + style flexibility |
| 15+ | ⚠️ Overfitting risk |

---

## 7 — Generate images

```zsh
# Generate 4 images with Auri persona
uv run generate.py --character auri --batch 4

# Specific scene type
uv run generate.py --character auri --type fashion --batch 4

# Fixed seed (reproducible)
uv run generate.py --character auri --batch 4 --seed 42

# Dry run (show prompt only, no image)
uv run generate.py --character auri --dry-run
```

Images saved to `outputs/<character>/`.

---

## 8 — Monitor training (GUI alternative)

> **Note:** `vendor/kohya_ss` is the **sd-scripts** repo (training scripts only) — it has no `gui.py`.
> The GUI lives in a separate repo [`bmaltais/kohya_ss`](https://github.com/bmaltais/kohya_ss).
> For most use cases **TensorBoard (step 6) is sufficient and already works.**

### Option A — TensorBoard (recommended, already installed)

```zsh
uv run tensorboard --logdir logs/lora_auri --port 6006
# Open http://localhost:6006
```

### Option B — Full kohya_ss GUI

```zsh
# Clone the GUI repo (one-time, ~500 MB)
git clone https://github.com/bmaltais/kohya_ss vendor/kohya_gui

# Install GUI deps
python3.11 -m venv vendor/kohya_gui/.venv
vendor/kohya_gui/.venv/bin/pip install -r vendor/kohya_gui/requirements.txt

# Launch
vendor/kohya_gui/.venv/bin/python vendor/kohya_gui/kohya_gui.py \
  --listen 0.0.0.0 --port 7860
# Open http://localhost:7860
```

---

## ComfyUI backend (optional, Windows)

The default backend is diffusers. ComfyUI is used only with `generate.py --comfyui`.

```powershell
# 1. Start the server (uses the project .venv and the model folders in configs/comfyui_model_paths.yaml)
powershell -File scripts/comfyui.ps1          # http://127.0.0.1:8188

# 2. In another terminal
uv run generate.py --character auri --type fashion --batch 1 --comfyui
```

- The only workflow is `workflows/luna_portrait.json` (plain RealVisXL text-to-image, node IDs hard-coded in `comfyui_client.py`). It has no LoRA or IP-Adapter node, so it will not reproduce a trained character.
- `comfyui.sh` is the old macOS launcher; use `scripts/comfyui.ps1` on Windows.
- Reference sheet (identity + angle) runs here, not in diffusers: `uv run scripts/ref_sheet_comfy.py` (about 17 s per 1024px view). It needs the `ComfyUI_IPAdapter_plus` custom node (`git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus` into `vendor/ComfyUI/custom_nodes/`, reviewed at commit a0f451a). The diffusers `scripts/ref_sheet.py` ControlNet path spills out of 12 GB and runs ~3 minutes per step, so only use it for IP-Adapter-only experiments.
- Face identity score: `uv run scripts/face_sim.py outputs/comfy_batch2 --baseline` (ArcFace R50 from `immich-app/buffalo_l` `recognition/model.onnx` saved as `models/face/arcface_r50.onnx`, plus YuNet `face_detection_yunet_2023mar.onnx` saved as `models/face/yunet.onnx`; CPU only). Read scores against the baseline row: rejected master-face candidates score 0.47-0.75 against the master, so generated heads at ~0.55 are a family resemblance, not an exact match.
- Content images for MODEL_001 (no LoRA; the v1/v2 LoRAs did not beat this recipe in an A/B, see `scripts/lora_ab_comfy.py`): `uv run scripts/generate_comfy.py --type casual_cool --batch 3` (about 17 s each, full-body 832x1216; add `--dry-run` to see the prompts, `--face-pass` to re-render the head at 1024px with a strong IP-Adapter, `--depth 0.5` for the hourglass depth map). It writes the image plus a post JSON with caption and tags to `outputs/comfy/`. Close GPU-heavy apps (games, Ollama) first: they make it several times slower.
- Body shape from a target image (MODEL_001): the Sora street collage `characters/model_001/target/sora_sheet.png` (git-ignored) sets body shape, skin tone and eye colour only, never the face. Steps:
  1. One-time: download Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) into `models/depth/depth-anything-v2-small`.
  2. `uv run scripts/depth_from_target.py` turns the walking panel into `characters/model_001/target/depth_walk.png`: background and floor are flattened so the scenery is not copied, `--head-scale` (default 0.8) shrinks the head about the neck, `--shoulder-scale` (1.08) widens the shoulders, `--bust-boost` (20) fills the chest, `--arm-grow` (default 4 px) fattens the arms, `--no-flat-bg` keeps the scenery.
  3. `uv run scripts/generate_comfy.py --type casual_cool --pose front --seed 80 --depth 0.9 --depth-end 0.8 --depth-image characters/model_001/target/depth_walk.png --face-pass --out outputs/comfy_body/auto`. `--depth-image` turns OpenPose off, so the pose is fixed to the guide (walking with a tote bag). It also runs `scripts/skin_balance.py` and saves `<stem>_final.png` with the arms, hands and neck matched to the face tone (`--no-skin-balance` skips it; the guide-based mask assumes the plain-wall pose of this guide).
  - Checks: `scripts/body_metrics.py <folder>` (skin tone of cheek/neck/forearm; its arm/face width ratio is noisy, judge arm size by eye) and `scripts/tune_skin_eyes.py <folder>` (re-renders the face pass with skin/eye prompt variants on existing stage-1 images).
  - Limits: IP-Adapter locks the Master Face skin colour, so prompts cannot add freckles or a pinker tone; bodies stay a little slimmer than the target; the guide's outfit is not fixed.
- Outfit from a reference photo: describe it as a content type in `CONTENT_CONFIGS` (`src/idol_generate/prompt_builder.py`); only the clothes are used, never the model's face. `--type boho_set` is an off-shoulder ruffle crop top with a tiered cream maxi skirt. Use it without `--depth`: the hourglass depth turns the skirt into trousers.
- Repeating an identical prompt and seed is fully cached by ComfyUI; the client adds a unique save prefix so it still returns an image.
- ComfyUI prints a warning that the frontend package is 1.42.10 (recommends 1.42.11); the API and UI work.

---

## Project structure

```
idol-generate/
├── generate.py              # inference entrypoint
├── train_lora.py            # training entrypoint
├── prepare_dataset.py       # BLIP captioning + dataset prep
├── scrape_images.py         # Instagram + site scraper
├── characters/              # persona JSON files
│   ├── luna.json
│   ├── auri.json
│   └── model_001.json       # synthetic character (see docs/MASTER_FACE_PLAN.md)
├── configs/                 # training configs
│   ├── lora_luna.toml
│   ├── lora_auri.toml       # old Mac-era config
│   └── lora_auri_v2.toml    # CUDA baseline
├── scripts/                 # audit, snapshot, experiment, eval, master-face tools
├── experiments/             # one-variable experiments (control/variant/results.md)
├── docs/                    # migration and master-face plans
├── tests/                   # pytest (run: uv run --with pytest pytest -q tests)
├── datasets/
│   ├── instagram/           # scraped raw images (gitignored)
│   └── lora/                # prepared training datasets
├── models/
│   ├── checkpoints/         # base model (gitignored)
│   └── loras/               # trained LoRA weights
├── logs/                    # TensorBoard logs + sample previews
├── outputs/                 # generated images
├── src/idol_generate/       # core library
│   ├── pipeline.py
│   ├── prompt_builder.py
│   ├── character.py
│   └── comfyui_client.py
└── vendor/kohya_ss/         # kohya_ss (gitignored)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `fp16/bf16 requires a GPU` | accelerate 1.6.0 blocks both on MPS | `mixed_precision = "no"` in toml |
| `model_index.json not found` | Model is a `.safetensors` file, not Diffusers folder | Point path to the `.safetensors` file directly |
| `MPS out of memory` | fp32 + SDXL 1024px exceeds 16 GB RAM | Use `resolution = "768,768"` |
| `No data found` | `train_data_dir` points to image folder, not parent | Set `train_data_dir` to parent of `5_auri_v1/` |
| `no kernel image is available for execution` | torch build has no sm_120 kernels (RTX 50-series) | Use torch 2.7.1+cu128 (pinned in `pyproject.toml`) |
| `UnicodeEncodeError` on emoji (Windows) | cp1252 console | `PYTHONUTF8=1`; `generate.py` also reconfigures its streams |
| `unrecognized arguments: --output_dir` | `train_lora.py` extra args are not passed as flags | Put the setting in a copy of the toml |
| `FileNotFoundError` mid-training | images were moved/deleted while training | Train from an immutable snapshot, not the live folder |
| `CUDA error: out of memory` during generation | another app holds GPU memory | Close GPU apps; VAE slicing/tiling is already on |
| Prompt details have no effect | SDXL text encoders read only 77 tokens each | Put the detail first; check with `count_tokens` in `prompt_builder.py` |
| `403 Forbidden` on Instagram | Session cookie expired or not logged in | Re-run `--extract-cookies` |
| `0 images found` on site | Site uses JavaScript rendering | Default Playwright mode handles this automatically |