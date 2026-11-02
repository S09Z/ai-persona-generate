# EXP-006-lora-weight

Base model: RealVisXL V4.0
Baseline config: configs/lora_auri_v2.toml
Variable changed: inference only
Dataset snapshot:
Training seed:
Optimizer steps:

## Training
| | Control | Variant |
|---|---|---|
| Final loss | | |
| Wall time | | |
| Peak VRAM | | |

## Scores (1-5; artifacts & memorization: lower is better)
| Metric | Control | Variant |
|---|---|---|
| Facial likeness | | |
| Eye/nose/jaw consistency | | |
| Identity across seeds | | |
| Prompt adherence | | |
| Facial artifacts | | |
| Training-image memorization | | |

Held-out references used: yes / no

## Conclusion
Best-performing arm:
Largest improvement:
Largest regression:
Decision for next experiment:

## Preliminary observations (read by Claude from contact sheets, 2026-11-02; not user-scored)
Run: auri_v2.safetensors (final, loss 0.119), weights 0.6/0.8/1.0/1.2 + base, 3 prompts x 3 seeds, 1024px, 25 steps.
- Identity: already clear at 0.6 (base shows a blonde Western face; every LoRA column shows the Auri look). Little extra likeness from 1.0 and 1.2.
- A_frontal: at 1.2 skin looks flatter and over-smooth, with white/grey hair streaks in two of three seeds.
- C_context: at 1.0-1.2 the prompt drifts (bandana/cap, torn jeans, jacket redesigned, garbled text on clothing); 0.6-0.8 follow the prompt (black jacket, jeans) best.
- Tentative: 0.6-0.8 is the usable range; keep 0.8 in characters/auri.json pending user scoring.
- Not checked: B_expression sheet, held-out reference comparison, memorization against training images.
