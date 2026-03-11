#!/usr/bin/env python3
"""
For each checkpoint subdirectory:
  - Read train_loss and val_loss from checkpoints/{subdir}/losses.json as denominators.
  - Find the matching model run (same config prefix; checkpoint uses -e1-, model uses -e1500-).
  - In that model run: scale all train_loss by train_denominator and val_loss by val_denominator
    in losses.json, and rename *-loss_{value}.pt to *-loss_{value/val_denominator}.pt.
"""

import argparse
import json
import re
from pathlib import Path


def get_config_prefix(subdir_name: str, marker: str) -> str | None:
    """Extract config prefix: part before marker (e.g. '-e1-' or '-e1500-')."""
    if marker not in subdir_name:
        return None
    return subdir_name.split(marker)[0] + marker


def find_model_subdir(model_dir: Path, checkpoint_subdir_name: str) -> Path | None:
    """Find model subdirectory that matches this checkpoint run (same config, e1500 variant)."""
    # Checkpoint names: ...-e1-TIMESTAMP. Model names: ...-e1500-p90-...
    prefix_e1 = get_config_prefix(checkpoint_subdir_name, "-e1-")
    if not prefix_e1:
        return None
    config_base = prefix_e1.removesuffix("-e1-")
    target_prefix = config_base + "-e1500-"
    for d in model_dir.iterdir():
        if d.is_dir() and d.name.startswith(target_prefix):
            return d
    return None


def normalize_losses(
    checkpoints_root: Path,
    model_dir: Path,
    dry_run: bool = False,
) -> None:
    checkpoints_root = checkpoints_root.resolve()
    model_dir = model_dir.resolve()

    for cp_subdir in sorted(checkpoints_root.iterdir()):
        if not cp_subdir.is_dir():
            continue

        cp_losses_path = cp_subdir / "losses.json"
        if not cp_losses_path.exists():
            print(f"Skip (no losses.json): {cp_subdir.name}")
            continue

        with open(cp_losses_path) as f:
            cp_losses = json.load(f)
        if not cp_losses or not isinstance(cp_losses, list):
            print(f"Skip (empty or invalid losses): {cp_subdir.name}")
            continue

        first = cp_losses[0]
        train_denominator = first.get("train_loss")
        val_denominator = first.get("val_loss")
        if train_denominator is None or val_denominator is None:
            print(f"Skip (missing train_loss/val_loss): {cp_subdir.name}")
            continue
        if train_denominator == 0 or val_denominator == 0:
            print(f"Skip (zero denominator): {cp_subdir.name}")
            continue

        model_subdir = find_model_subdir(model_dir, cp_subdir.name)
        if model_subdir is None:
            print(f"Skip (no matching model dir): {cp_subdir.name}")
            continue

        model_losses_path = model_subdir / "losses.json"
        if not model_losses_path.exists():
            print(f"Skip (no model losses.json): {model_subdir.name}")
            continue

        with open(model_losses_path) as f:
            model_losses = json.load(f)
        if not isinstance(model_losses, list):
            print(f"Skip (model losses not a list): {model_subdir.name}")
            continue

        # Scale losses
        normalized = []
        for entry in model_losses:
            t = entry.get("train_loss")
            v = entry.get("val_loss")
            normalized.append({
                "train_loss": t / train_denominator if t is not None else None,
                "val_loss": v / val_denominator if v is not None else None,
            })

        if not dry_run:
            with open(model_losses_path, "w") as f:
                json.dump(normalized, f, indent=2)
        print(f"Normalized losses: {model_subdir.name} (train/={train_denominator}, val/={val_denominator})")

        # Rename *-loss_{value}.pt -> *-loss_{value/val_denominator}.pt
        loss_pt_pattern = re.compile(r"^(.+)-loss_([0-9.eE+-]+)\.pt$")
        for pt_path in model_subdir.iterdir():
            if not pt_path.is_file() or pt_path.suffix != ".pt":
                continue
            m = loss_pt_pattern.match(pt_path.name)
            if not m:
                continue
            prefix, old_value_str = m.group(1), m.group(2)
            try:
                old_value = float(old_value_str)
            except ValueError:
                continue
            new_value = old_value / val_denominator
            new_name = f"{prefix}-loss_{new_value}.pt"
            new_path = pt_path.parent / new_name
            if new_path == pt_path:
                continue
            if not dry_run:
                pt_path.rename(new_path)
            print(f"  Rename: {pt_path.name} -> {new_name}")


def main():
    p = argparse.ArgumentParser(description="Normalize model losses and checkpoint filenames by checkpoint denominators.")
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"), help="Checkpoints root (default: checkpoints)")
    p.add_argument("--model-dir", type=Path, default=Path("models"), help="Model (or model_cfg) root (default: models)")
    p.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = p.parse_args()

    normalize_losses(args.checkpoints, args.model_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
