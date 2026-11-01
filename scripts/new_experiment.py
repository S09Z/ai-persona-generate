"""
scripts/new_experiment.py -- Scaffold a one-variable LoRA training experiment.

Creates experiments/<name>/ with a control and a variant config copied from a
baseline toml. Exactly the --set overrides differ between them; everything
else (seed, dataset, steps) stays identical so the comparison is attributable.

Usage:
    uv run scripts/new_experiment.py EXP-007-network-dropout \
        --baseline configs/lora_auri_v2.toml \
        --set lora.network_dropout=0.10

    # then train each arm:
    uv run train_lora.py --config experiments/EXP-007-network-dropout/control/config.toml
    uv run train_lora.py --config experiments/EXP-007-network-dropout/variant/config.toml

For inference-only experiments (e.g. LoRA weight) use --no-variant.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

RESULTS_TEMPLATE = """# {name}

Base model: RealVisXL V4.0
Baseline config: {baseline}
Variable changed: {change}
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
"""


def set_value(text: str, section: str, key: str, value: str) -> str:
    """Set key=value inside [section]; uncomment an existing `# key` line if present."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == f"[{section}]"), None)
    if start is None:
        sys.exit(f"section [{section}] not found in baseline config")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    pat = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=")
    rendered = value if re.fullmatch(r"-?[\d.]+(e-?\d+)?|true|false|\".*\"", value) else f'"{value}"'
    for i in range(start + 1, end):
        if pat.match(lines[i]):
            lines[i] = f"{key} = {rendered}"
            return "\n".join(lines) + "\n"
    lines.insert(end, f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def make_arm(base: str, exp_dir: Path, arm: str, overrides: list[str]) -> None:
    text = base
    arm_dir = exp_dir / arm
    arm_dir.mkdir(parents=True)
    for key, val in (
        ("output.output_dir", f'"{(arm_dir / "model").as_posix()}"'),
        ("output.output_name", f'"{exp_dir.name}-{arm}"'),
        ("output.character_json", '""'),
        ("logging.logging_dir", f'"{(arm_dir / "logs").as_posix()}"'),
        ("sample.sample_every_n_epochs", "0"),
    ):
        sec, k = key.split(".")
        text = set_value(text, sec, k, val)
    for ov in overrides:
        key, val = ov.split("=", 1)
        sec, k = key.split(".")
        text = set_value(text, sec, k, val)
    (arm_dir / "config.toml").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="e.g. EXP-007-network-dropout")
    ap.add_argument("--baseline", default="configs/lora_auri_v2.toml")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value",
                    help="the single change applied to the variant arm")
    ap.add_argument("--no-variant", action="store_true", help="inference-only experiment")
    args = ap.parse_args()

    exp_dir = Path("experiments") / args.name
    if exp_dir.exists():
        sys.exit(f"{exp_dir} already exists")
    if not args.no_variant and not args.set:
        sys.exit("give the variant's change with --set section.key=value (or use --no-variant)")

    base = Path(args.baseline).read_text(encoding="utf-8")
    exp_dir.mkdir(parents=True)
    (exp_dir / "evaluation").mkdir()
    if not args.no_variant:
        make_arm(base, exp_dir, "control", [])
        make_arm(base, exp_dir, "variant", args.set)
    shutil.copy(args.baseline, exp_dir / "baseline.toml")
    (exp_dir / "results.md").write_text(
        RESULTS_TEMPLATE.format(name=args.name, baseline=args.baseline,
                                change=", ".join(args.set) or "inference only"),
        encoding="utf-8",
    )
    print(f"created {exp_dir}")


if __name__ == "__main__":
    main()
