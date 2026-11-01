# MODEL_001: Master Face and Character Reference Sheet plan

Goal: a locked, synthetic Thai face (no real-person photos) -> reference sheet -> consistent
dataset -> `model_001` LoRA -> fashion content. Character spec: `characters/model_001.json`.

## Findings (2026-11-01)
- Installed: SDXL RealVisXL V4.0, diffusers 0.37.1 (native SDXL IP-Adapter support), peft, kohya trainer, eval_grid.
- Not installed: IP-Adapter weights, CLIP image encoder, ControlNet (openpose), insightface. `vendor/ComfyUI/models/*` folders are empty.
- Disk: 178 GB free on D:. GPU is busy until the Auri v2 training finishes (~1h15m left at last check).
- Recommendation: do identity work in-repo with diffusers IP-Adapter (reuses `pipeline.py`, scriptable, seed-logged), not ComfyUI.

## Stages

### 0. Prerequisites (after training ends)
- Download: `h94/IP-Adapter` `sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors` (~1 GB) and its image encoder `models/image_encoder` (~2.5 GB).
- Download: an SDXL OpenPose ControlNet (e.g. `thibaud/controlnet-openpose-sdxl-1.0` or `xinsir/controlnet-openpose-sdxl-1.0`, ~2.5 GB) for pose control in stage 3.
- Face-similarity metric: try `insightface` (buffalo_l, needs onnxruntime-gpu; Windows build can fail). Fallback: manual review + CLIP image similarity.

### 1. Face exploration (no identity model, text only)
`scripts/master_face_candidates.py` (to write): 6 prompt variants x 8 fixed seeds = 48 headshots, 1024px, 25 steps, base RealVisXL, no LoRA.
- Variants vary only soft details inside the locked choices (soft oval, round eyes, long dark hair, Korean-Thai): eye spacing, nose bridge, lip fullness, brow shape, expression, hair parting.
- Prompt wording comes from `character_bible`; no celebrity or real-person names.
- Output: contact sheets in `characters/model_001/master_face/candidates/` + `settings.json`.
- Time: ~30 min of GPU.

### 2. Lock the Master Face (your decision)
- Pick 1 (shortlist 3). Save `master_face.png`, plus seed, prompt, settings in `master_face.json`.
- Checks: adult-looking 25-29, honey skin tone, Thai look reads naturally, not a recognizable real person (do a reverse-image check yourself), works at angles when imagined.
- After locking, never edit these fixed fields: face, skin tone, hair, body proportions.

### 3. Character Reference Sheet
`scripts/ref_sheet.py` (to write): IP-Adapter Plus Face conditioned on `master_face.png`, ControlNet OpenPose for angle.
- Head views: front, 3/4 left, 3/4 right, profile left/right (neutral); smile; calm.
- Body views: full-body front, side, back; plus 4 fashion archetypes (minimalist, modern executive, soft elegant, evening muse).
- Fixed seeds per view; assemble one sheet (PNG) + individual files.
- Acceptance: same person across views by eye and by similarity score >= threshold set from the first run.

### 4. Synthetic dataset (30-60 images)
- Generate from the sheet with varied outfits/backgrounds from the palette rules; archetype coverage evenly spread.
- Filter: similarity score, anatomy check, no logos/text. Tag `view` in manifest (frontal / 3-4 / profile) so EXP-005 is possible.
- Hold out 8-10 images (never trained on). Build with `scripts/build_clean_snapshot.py`.

### 5. Train `model_001_v1`
- Copy `configs/lora_auri_v2.toml`, set snapshot path, trigger e.g. `m001_v1`, output name `model_001_v1`.
- Evaluate with `scripts/eval_grid.py` (viewpoint prompt set, but swap in the new trigger) vs the IP-Adapter-only baseline.
- Only then set `lora.path` / `lora.trigger_word` in `model_001.json`.

### 6. Content integration
- Replace `daily_life/fashion/selfie/story` in `prompt_builder.py` with the 4 archetypes; read `character_bible.production_rules`.
- Keep prompt_1 under 77 tokens; push detail to prompt_2.
- AI-generated disclosure before publishing (Shopee/Meta rules).

## Decisions needed
1. IP-Adapter in diffusers (recommended) vs ComfyUI workflows (PhotoMaker folder exists but nothing installed).
2. Approve ~6-7 GB of model downloads.
3. Eye color (dark brown is an assumption) and expression direction for the candidate set.

## Risks
| Risk | Mitigation |
|---|---|
| Identity drift at profile/back views | ControlNet pose + similarity filter; drop drifting images |
| Candidate resembles a real person | reverse-image check before locking; no named references |
| Synthetic-only data bakes in AI artifacts | strong manual filter, 30-60 curated images, low repeat count |
| insightface fails to install | CLIP similarity + manual review |
| GPU contention | start after Auri v2 training finishes |
