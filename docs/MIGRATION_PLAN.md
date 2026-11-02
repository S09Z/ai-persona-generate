# Migration plan: Mac/MPS → Windows 11 / RTX 5070 (12GB) + experiment workflow

Target stack: Windows 11, Python 3.11, RealVisXL V4.0, kohya sd-scripts, RTX 5070 (Blackwell, sm_120).
Reference: weekly LoRA experiment series (#5 viewpoint, #6 LoRA weight, #7 network dropout).

## Current state (verified 2026-11-01)

| Area | State |
|---|---|
| Main env (`.venv`) | torch 2.7.1+cu128, diffusers 0.37.1, peft 0.21.2. GPU runs (98% util). |
| LoRA at inference | Now loaded in `pipeline.py` (`--lora-weight`, `--no-lora`). Untested end to end. |
| Test generation | 832x1216, 25 steps ran >10 min at 11.4/12 GB VRAM: almost certainly spilling to shared memory. |
| Training env | `train_lora.py` expects `vendor/kohya_ss/.venv`, which does not exist. `.venv-train` exists (torch 2.12 nightly cu128, has sm_120). |
| Training config | Still Mac-era: 768px, comments about MPS, toml says bf16 but README says "no". |
| Dataset | 148 images, 132 captions, 21 `(1)` duplicates, raw BLIP captions. |
| Checkpoints | `auri_v1-000002/000004` only. Prior log shows `avr_loss=nan`, killed. |
| Secrets | Cookie file untracked, but still in git history and session still valid. |
| Uncommitted | .gitignore, README, auri.json, generate.py, pipeline.py, train_lora.py, pyproject.toml, uv.lock, prompts rename, cookie untrack. |

## Phase 0: Safety and baseline (30 min)
1. Revoke the Instagram session (log out everywhere). Optionally purge history with `git filter-repo`.
2. Commit current work in two commits: (a) LoRA loading + path/prompt fixes, (b) torch cu128 + lockfile.
3. Smoke test ComfyUI from the workspace after the torch bump (`./comfyui.sh` equivalent on Windows). If broken, move ComfyUI to its own venv.

## Phase 1: Inference on CUDA (1 hr)
1. Fix generation speed: add `pipe.enable_vae_slicing()` / `enable_vae_tiling()`; fall back to `enable_model_cpu_offload()` if still >11 GB. Consider 832x1216 → 768x1152 for tests.
2. Windows console: set `PYTHONUTF8=1` in `generate.py` (or reconfigure stdout) to stop the emoji `UnicodeEncodeError`.
3. Remove MPS-only branches/comments; keep a clean `cuda` path. Use SDPA (default) instead of xformers.
4. Verify: one image with `--no-lora`, one with LoRA at 0.8, same seed → visibly different.
Exit: both images generate in under ~1 min each.

## Phase 2: Training environment (1 hr)
1. Decide trainer env: reuse `.venv-train` (torch nightly, works today) or build `vendor/kohya_ss/.venv` with torch 2.7.1+cu128. Recommend the stable one for reproducibility.
2. Point `KOHYA_PYTHON` in `train_lora.py` at the chosen env (or make it configurable via `--python`).
3. Install kohya requirements + `bitsandbytes` (Windows build) / `tensorboard`.
4. Dry run: `uv run train_lora.py --config configs/lora_auri.toml --dry-run`, then 10-step real run.

## Phase 3: Config migration (30 min)
Change `configs/lora_auri.toml` (copy to `lora_auri_v2.toml`, keep the old one):
- `resolution` 1024, `mixed_precision` bf16, `gradient_checkpointing` true, batch 1 (12 GB).
- Remove MPS comments; fix header math (steps = images x repeats x epochs / batch / accum).
- Drop hardcoded `--clip_skip 2` for SDXL (make configurable, default unset).
- Optional: `cache_latents`, `cache_text_encoder_outputs` (only if `network_train_unet_only`), 8-bit AdamW to save VRAM.
- Expose `network_dropout`, `rank_dropout`, `module_dropout`, `seed` in `build_command` for EXP-007.
- Write the resolved config + command into each run folder (reproducibility).

## Phase 4: Dataset cleanup (half day)
1. Dedupe the 21 `(1)` files (perceptual hash), write `manifest.csv` (file, source, hash, view: frontal/3-4/profile, keep/drop).
2. Caption the 16 uncaptioned images; replace "arafed/araffe" artifacts; keep trigger `auri_v1` first.
3. Hold out 10-15 images as reference only (never trained on).
4. Snapshot as `datasets/lora/auri_v1_snapshot_YYYYMMDD`; experiments reference snapshots, not the live folder.
5. Review licensing/consent for the scraped source before publishing the persona.

## Phase 5: Experiment harness (half day)
Create:
```
experiments/EXP-00N-<name>/{control,variant}/{config.toml,training.log}
experiments/EXP-00N-<name>/evaluation/  results.md  manifest.csv
```
- `scripts/eval_grid.py`: fixed prompts A/B/C x seeds (42, 123, 777, 2026) x LoRA/weight list; saves images + `settings.json` + contact sheet.
- `scripts/new_experiment.py`: scaffolds folder and copies the baseline config with a single named change.
- Log final loss, steps, peak VRAM, wall time to `results.md`.

## Phase 6: Run order
1. **Baseline**: retrain `auri_v1` on cleaned dataset at 1024 (confirm loss is not nan).
2. **EXP-006 (weight 0.6/0.8/1.0/1.2)**: no retraining, run first to set inference weight.
3. **EXP-005 (viewpoint)**: after manifest tags exist.
4. **EXP-007 (network dropout 0.10)**: two runs, same seed/steps.
Decide each next experiment from the observed failure pattern, as in the source series.

## Phase 7: Code cleanup (parallel, low risk)
- `prompt_builder.py`: single `mood` draw; shorten `prompt_1` below 77 tokens (move detail to `prompt_2`).
- Remove dead files: `idol-env`, extra `anime_workflow_*.json` variants if unused, tracked `logs/**/events.out.tfevents*`.
- README: Windows setup, new commands, correct checkpoint names.
- Add tests: `build_command` snapshot test, `Character.load`, prompt length check.

## Risks
| Risk | Mitigation |
|---|---|
| 12 GB VRAM at 1024 training | grad checkpointing, 8-bit optimizer, fall back to 896 |
| Nightly torch in `.venv-train` | pin stable 2.7.1+cu128 for trainer |
| ComfyUI breaks with torch bump | isolate its venv |
| Results not comparable across runs | snapshots + fixed seeds + saved configs |
| Cookie in git history | revoke session; optionally rewrite history |
