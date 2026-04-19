# idol-generate

AI Idol image generation pipeline — scrape reference photos, train a LoRA, generate images.

---

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Apple Silicon Mac (MPS) **or** NVIDIA GPU (CUDA)
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
├── auri_v1-epoch-02.safetensors  ← saved every 2 epochs
├── auri_v1-epoch-04.safetensors
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

## Project structure

```
idol-generate/
├── generate.py              # inference entrypoint
├── train_lora.py            # training entrypoint
├── prepare_dataset.py       # BLIP captioning + dataset prep
├── scrape_images.py         # Instagram + site scraper
├── characters/              # persona JSON files
│   ├── luna.json
│   └── auri.json
├── configs/                 # training configs
│   ├── lora_luna.toml
│   └── lora_auri.toml
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
| `403 Forbidden` on Instagram | Session cookie expired or not logged in | Re-run `--extract-cookies` |
| `0 images found` on site | Site uses JavaScript rendering | Default Playwright mode handles this automatically |