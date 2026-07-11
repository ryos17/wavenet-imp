from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.utils.prune as prune

from utils.util import build_dataloader, log_message, pre_emphasis, read_audio_mono, set_seed
from utils.wavenet import WaveNet


PrunableParam = Tuple[torch.nn.Module, str]


def _collect_prunable_params(model: torch.nn.Module) -> List[PrunableParam]:
    params: List[PrunableParam] = []
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if "weight" in name and param.requires_grad and param.dim() > 1:
                params.append((module, name))
    return params


def _global_sparsity(params_to_prune: List[PrunableParam]) -> float:
    total = 0
    zeros = 0
    for module, name in params_to_prune:
        weight = getattr(module, name)
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    if total == 0:
        return 0.0
    return float(zeros) / float(total)


def _global_zero_and_total(params_to_prune: List[PrunableParam]) -> Tuple[int, int]:
    total = 0
    zeros = 0
    for module, name in params_to_prune:
        weight = getattr(module, name)
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    return zeros, total


def _apply_one_shot_pruning(
    model: torch.nn.Module,
    params_to_prune: List[PrunableParam],
    target_sparsity: float,
    prune_type: str,
) -> float:
    del model
    if target_sparsity <= 0.0:
        return _global_sparsity(params_to_prune)

    if prune_type == "global":
        current_zeros, total = _global_zero_and_total(params_to_prune)
        target_zeros = min(max(math.ceil(target_sparsity * total), 0), total)
        additional_zeros = target_zeros - current_zeros
        if additional_zeros > 0:
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=additional_zeros,
            )
        return _global_sparsity(params_to_prune)

    if prune_type == "local":
        for module, name in params_to_prune:
            weight = getattr(module, name)
            total = int(weight.numel())
            current_zeros = int((weight == 0).sum().item())
            target_zeros = min(max(math.ceil(target_sparsity * total), 0), total)
            additional_zeros = target_zeros - current_zeros
            if additional_zeros > 0:
                prune.l1_unstructured(module, name=name, amount=additional_zeros)
        return _global_sparsity(params_to_prune)

    raise ValueError(f"Unsupported prune_type: {prune_type}")


def run_one_shot_from_checkpoint(
    checkpoint_path: str,
    sparsity_targets: List[float],
    prune_type: str,
    device_name: str | None,
    val_audio_seconds_override: int | None,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = ckpt["model_cfg"]
    ckpt_train_cfg = ckpt.get("train_cfg", {})
    seed = int(ckpt_train_cfg.get("seed", 42))

    input_wav = ckpt_train_cfg.get("input_wav")
    target_wav = ckpt_train_cfg.get("target_wav")
    chunk_size = int(ckpt_train_cfg.get("chunk_size", 4800))
    batch_size = int(ckpt_train_cfg.get("batch_size", 40))
    num_workers = int(ckpt_train_cfg.get("num_workers", 0))
    train_split = float(ckpt_train_cfg.get("train_split", 0.8))
    val_split = float(ckpt_train_cfg.get("val_split", 0.2))
    val_audio_seconds = int(val_audio_seconds_override if val_audio_seconds_override is not None else ckpt_train_cfg.get("val_audio_seconds", 5))
    device = torch.device(device_name if device_name else ckpt_train_cfg.get("device", "cuda"))

    if not input_wav or not target_wav:
        raise ValueError("Checkpoint train_cfg must contain input_wav and target_wav.")

    # Match old script behavior as closely as possible.
    set_seed(seed)

    # Create output directory and logger.
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    sparsity_tag = "-".join(f"{int(v * 100)}" for v in sparsity_targets)
    ckpt_stem = Path(checkpoint_path).stem
    run_name = f"{ckpt_stem}-plist{sparsity_tag}-{prune_type}-oneshot-{run_stamp}"
    save_dir = Path("checkpoints") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "logs.txt"

    log_message(log_path, f"Loaded checkpoint: {checkpoint_path}")
    log_message(log_path, f"Sparsity targets: {sparsity_targets}")
    log_message(log_path, f"Prune type: {prune_type}")
    log_message(log_path, f"Seed: {seed}")

    # Build validation set exactly like training scripts.
    x, sr_x = read_audio_mono(input_wav)
    y, sr_y = read_audio_mono(target_wav)
    if sr_x != sr_y:
        raise ValueError(f"Input SR ({sr_x}) != target SR ({sr_y}).")
    split_total = train_split + val_split
    if split_total <= 0.0:
        raise ValueError("train_split + val_split must be > 0.")
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    n_train = int(n * (train_split / split_total))
    n_train = min(max(n_train, chunk_size), n - chunk_size)
    x_val = x[n_train:]
    y_val = y[n_train:]

    val_loader = build_dataloader(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    # Base model from checkpoint.
    base_model = WaveNet.from_config_dict(model_cfg).to(device)
    base_model.load_state_dict(ckpt["model_state_dict"])
    dense_state = copy.deepcopy(base_model.state_dict())
    rf = base_model.receptive_field
    valid_start = max(0, rf - 1)
    mse = torch.nn.MSELoss()

    summary_rows: list[dict[str, float | str]] = []
    total_params = sum(p.numel() for p in base_model.parameters())
    train_start = datetime.now()

    for target_sparsity in sparsity_targets:
        model = WaveNet.from_config_dict(model_cfg).to(device)
        model.load_state_dict(dense_state)
        params_to_prune = _collect_prunable_params(model)
        total_prunable = sum(getattr(module, name).numel() for module, name in params_to_prune)
        before = _global_sparsity(params_to_prune)
        after = _apply_one_shot_pruning(model, params_to_prune, target_sparsity, prune_type)

        model.eval()
        val_loss_recomputed = float("nan")
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                pred_valid = pred[:, valid_start:]
                y_valid = yb[:, valid_start:]
                numerator = torch.sum((y_valid - pred_valid) ** 2)
                denominator = torch.sum(y_valid ** 2) + 1e-12
                esr_value = numerator / denominator
                val_losses.append(float(esr_value.item()))
        val_loss_recomputed = float(np.mean(val_losses)) if val_losses else float("nan")
        # For p00, use the checkpoint's stored best_val_loss to match old logs exactly.
        if abs(target_sparsity) < 1e-12 and "best_val_loss" in ckpt:
            val_loss = float(ckpt["best_val_loss"])
        else:
            val_loss = val_loss_recomputed

        target_tag = f"p{int(round(target_sparsity * 100)):02d}"
        target_dir = save_dir / target_tag
        target_dir.mkdir(parents=True, exist_ok=True)
        ckpt_name = f"{run_name}-{target_tag}-loss_{val_loss:.5e}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "sample_rate": sr_x,
                "sparsity_target": target_sparsity,
                "val_loss": val_loss,
                "source_checkpoint": checkpoint_path,
            },
            target_dir / ckpt_name,
        )

        num_samples = min(val_audio_seconds * sr_x, x_val.shape[0], y_val.shape[0])
        sample_x = x_val[:num_samples].copy()
        sample_y = y_val[:num_samples].copy()
        with torch.no_grad():
            sample_in = torch.from_numpy(sample_x).to(device)[None, :]
            sample_pred = model(sample_in).squeeze(0).detach().cpu().numpy().astype(np.float32)
        sf.write(str(target_dir / "source.wav"), sample_x, sr_x)
        sf.write(str(target_dir / "target.wav"), sample_y, sr_x)
        sf.write(str(target_dir / "model_output.wav"), sample_pred, sr_x)

        total_pruned = sum((getattr(module, name) == 0).sum().item() for module, name in params_to_prune)
        log_message(
            log_path,
            f"One-shot prune {target_tag} - val_loss={val_loss:.5e} - target={target_sparsity:.6f} - "
            f"before={before:.6f} - after={after:.6f} - "
            f"pruned/prunable={total_pruned/total_prunable:.6f} - pruned/total={total_pruned/total_params:.6f} - "
            f"val_loss_recomputed={val_loss_recomputed:.5e} - "
            f"total parameters: {total_params} - total prunable parameters: {total_prunable} - total pruned parameters: {total_pruned}"
        )
        summary_rows.append(
            {
                "target_sparsity": target_sparsity,
                "achieved_sparsity": after,
                "val_loss": val_loss,
                "checkpoint": str(target_dir / ckpt_name),
            }
        )

    with (save_dir / "prune_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    elapsed_s = (datetime.now() - train_start).total_seconds()
    log_message(log_path, f"Completed one-shot pruning sweep in {elapsed_s:.2f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot prune from an existing checkpoint.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/old/ch16_ungated-2026-03-01_13-44-09-625757/ch16_ungated-best.pt",
        help="Path to dense checkpoint to prune.",
    )
    parser.add_argument(
        "--sparsity_target",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        help="One or more target sparsities.",
    )
    parser.add_argument(
        "--prune_type",
        type=str,
        default="local",
        help="Pruning type: global or local.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--val_audio_seconds",
        type=int,
        default=None,
        help="Optional validation audio export length override.",
    )
    args = parser.parse_args()

    sparsity_targets = sorted({float(v) for v in args.sparsity_target})
    prune_type = str(args.prune_type).lower()
    if prune_type not in {"global", "local"}:
        raise ValueError("prune_type must be one of {'global', 'local'}.")

    run_one_shot_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        sparsity_targets=sparsity_targets,
        prune_type=prune_type,
        device_name=args.device,
        val_audio_seconds_override=args.val_audio_seconds,
    )


if __name__ == "__main__":
    main()
