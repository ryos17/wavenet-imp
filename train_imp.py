from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.utils.prune as prune

from utils.util import (
    build_dataloader,
    load_json,
    log_message,
    pre_emphasis,
    read_audio_mono,
    set_seed,
)
from utils.wavenet import WaveNet


PrunableParam = Tuple[torch.nn.Module, str]


def _collect_prunable_params(model: torch.nn.Module) -> List[PrunableParam]:
    """Collect prunable parameters from the model."""
    params: List[PrunableParam] = []
    # Go through all modules and collect prunable parameters
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if (
                "weight" in name
                and param.requires_grad
                and param.dim() > 1
            ):
                params.append((module, name))
    return params


def _tensor_sparsity(tensor: torch.Tensor) -> float:
    """Compute sparsity of a tensor."""
    if tensor.numel() == 0:
        return 0.0
    return float((tensor == 0).sum().item()) / float(tensor.numel())


def _global_sparsity(params_to_prune: List[PrunableParam]) -> float:
    """Compute global sparsity from non-zero counts on selected parameters."""
    total = 0
    zeros = 0
    # Go through all prunable parameters and count zeros
    for module, name in params_to_prune:
        weight = getattr(module, name)
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    if total == 0:
        return 0.0
    return float(zeros) / float(total)


def _global_zero_and_total(params_to_prune: List[PrunableParam]) -> Tuple[int, int]:
    """Return global zero-count and total-count across prunable parameters."""
    total = 0
    zeros = 0
    for module, name in params_to_prune:
        weight = getattr(module, name)
        total += weight.numel()
        zeros += int((weight == 0).sum().item())
    return zeros, total


def _scheduled_sparsity(
    global_step: int,
    steps_per_epoch: int,
    prune_start_epoch: int,
    prune_end_epoch: int,
    sparsity_target: float,
    prune_schedule: str,
) -> float:
    """Compute scheduled sparsity based on the global step."""
    start_step = (prune_start_epoch - 1) * steps_per_epoch
    end_step_exclusive = prune_end_epoch * steps_per_epoch
    
    # Return 0.0 if before the start step, or the target sparsity if after the end step
    if global_step < start_step:
        return 0.0
    if global_step >= end_step_exclusive:
        return sparsity_target
    
    # Compute progress through the pruning schedule
    total_prune_steps = end_step_exclusive - start_step
    progress = float((global_step - start_step) + 1) / float(total_prune_steps)
    progress = min(max(progress, 0.0), 1.0)
    
    # Compute scheduled sparsity based on the prune schedule
    if prune_schedule == "linear":
        return sparsity_target * progress
    if prune_schedule == "exponential":
        # Exponential sparsity growth: density decays exponentially to final target.
        return 1.0 - ((1.0 - sparsity_target) ** progress)
    raise ValueError(f"Unsupported prune_schedule: {prune_schedule}")


def _apply_incremental_pruning(
    model: torch.nn.Module,
    params_to_prune: List[PrunableParam],
    target_sparsity: float,
    prune_type: str,
) -> float:
    """Apply incremental pruning to the model."""
    del model

    # Return current sparsity if target sparsity is 0.0
    if target_sparsity <= 0.0:
        return _global_sparsity(params_to_prune)
    
    # Apply global pruning
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

    # Apply local pruning
    if prune_type == "local":
        for module, name in params_to_prune:
            weight = getattr(module, name)
            total = int(weight.numel())
            current_zeros = int((weight == 0).sum().item())
            target_zeros = min(max(math.ceil(target_sparsity * total), 0), total)
            additional_zeros = target_zeros - current_zeros
            if additional_zeros <= 0:
                continue
            prune.l1_unstructured(module, name=name, amount=additional_zeros)
        return _global_sparsity(params_to_prune)

    raise ValueError(f"Unsupported prune_type: {prune_type}")


def train(model_cfg: Dict, train_cfg: Dict) -> None:
    # Load train config
    seed = int(train_cfg["seed"])
    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    epochs = int(train_cfg["epochs"])
    input_wav = train_cfg["input_wav"]
    target_wav = train_cfg["target_wav"]
    chunk_size = int(train_cfg["chunk_size"])
    num_workers = int(train_cfg["num_workers"])
    train_split = float(train_cfg["train_split"])
    val_split = float(train_cfg["val_split"])
    val_audio_seconds = int(train_cfg["val_audio_seconds"])
    device_name = train_cfg["device"]
    sparsity_target = float(train_cfg["sparsity_target"])
    prune_type = str(train_cfg["prune_type"]).lower()
    prune_schedule = str(train_cfg["prune_schedule"]).lower()
    prune_start_epoch = int(train_cfg["prune_start_epoch"])
    prune_end_epoch = int(train_cfg["prune_end_epoch"])

    # Validate pruning parameters
    if not (0.0 <= sparsity_target < 1.0):
        raise ValueError("sparsity_target must be in [0.0, 1.0).")
    if prune_type not in {"global", "local"}:
        raise ValueError("prune_type must be one of {'global', 'local'}.")
    if prune_schedule not in {"linear", "exponential"}:
        raise ValueError("prune_schedule must be one of {'linear', 'exponential'}.")
    if prune_start_epoch < 1 or prune_end_epoch < prune_start_epoch or prune_end_epoch > epochs:
        raise ValueError("Require 1 <= prune_start_epoch <= prune_end_epoch <= epochs.")

    # Set seed and device
    set_seed(seed)
    device = torch.device(device_name)

    # Create save directory with model config basename and date
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    wav_name = Path(target_wav).stem
    run_name = f"{wav_name}-b{batch_size}-lr{learning_rate}-e{epochs}-p{int(sparsity_target * 100)}-{prune_type}-{prune_schedule}-ps{prune_start_epoch}-pe{prune_end_epoch}-{run_stamp}"
    save_dir = Path("checkpoints") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = save_dir / "logs.txt"
    losses_json_path = save_dir / "losses.json"
    log_message(log_path, f"Train config: {train_cfg}")
    log_message(log_path, f"Model config: {model_cfg}")
    train_start_time = datetime.now()

    # Load audio files
    x, sr_x = read_audio_mono(input_wav)
    y, sr_y = read_audio_mono(target_wav)
    if sr_x != sr_y:
        raise ValueError(f"Input SR ({sr_x}) != target SR ({sr_y}).")
    split_total = train_split + val_split
    if split_total <= 0.0:
        raise ValueError("train_split + val_split must be > 0.")

    # Load model, optimizer and loss function
    model = WaveNet.from_config_dict(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse = torch.nn.MSELoss()
    rf = model.receptive_field
    valid_start = max(0, rf - 1)

    # Collect prunable parameters
    params_to_prune = _collect_prunable_params(model)
    total_prunable_params = sum(getattr(module, name).numel() for module, name in params_to_prune)
    total_params = sum(p.numel() for p in model.parameters())
    log_message(log_path, "=============================================================")
    log_message(log_path, f"Found {total_prunable_params} prunable parameters out of {total_params} total parameters.")
    log_message(log_path, "=============================================================")
    if not params_to_prune:
        raise ValueError("No prunable parameters found (expected weight tensors with dim > 1).")
    
    # Align audio files
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    # Split audio files into train and validation sets
    n_train = int(n * (train_split / split_total))
    n_train = min(max(n_train, chunk_size), n - chunk_size)
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:]
    y_val = y[n_train:]

    # Build data loaders
    train_loader = build_dataloader(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )
    val_loader = build_dataloader(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )
    train_steps_per_epoch = len(train_loader)

    # Train loop
    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_basename: str | None = None
    epoch_loss_history: list[dict[str, float | None]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        current_sparsity = _global_sparsity(params_to_prune)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            # Compute scheduled sparsity
            global_step = (epoch - 1) * train_steps_per_epoch + batch_idx
            target_batch_sparsity = _scheduled_sparsity(
                global_step=global_step,
                steps_per_epoch=train_steps_per_epoch,
                prune_start_epoch=prune_start_epoch,
                prune_end_epoch=prune_end_epoch,
                sparsity_target=sparsity_target,
                prune_schedule=prune_schedule,
            )
            # Apply incremental pruning
            current_sparsity = _apply_incremental_pruning(
                model=model,
                params_to_prune=params_to_prune,
                target_sparsity=target_batch_sparsity,
                prune_type=prune_type,
            )

            # Move data to device
            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            pred = model(xb)

            # Compute loss
            pred_valid = pred[:, valid_start:]
            y_valid = yb[:, valid_start:]
            pred_pre = pre_emphasis(pred_valid)
            y_pre = pre_emphasis(y_valid)
            loss = mse(pred_pre, y_pre)

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Track batch loss
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)

        # Compute average train loss
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                # Move data to device
                xb = xb.to(device)
                yb = yb.to(device)

                # Forward pass
                pred = model(xb)

                # Compute loss
                pred_valid = pred[:, valid_start:]
                y_valid = yb[:, valid_start:]
                pred_pre = pre_emphasis(pred_valid)
                y_pre = pre_emphasis(y_valid)
                val_loss = mse(pred_pre, y_pre)

                # Track batch validation loss
                val_loss_value = float(val_loss.item())
                val_losses.append(val_loss_value)

        # Compute average validation loss
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        # Print epoch results
        log_message(log_path, f"Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.5e} - val_loss={avg_val_loss:.5e} - current_sparsity={current_sparsity:.2f}")

        # Update best model and must be fully pruned
        if avg_val_loss < best_val_loss and current_sparsity >= sparsity_target:
            # Delete previous best checkpoint if it exists
            if best_ckpt_basename is not None:
                prev_best_save_ckpt = save_dir / best_ckpt_basename
                if prev_best_save_ckpt.exists():
                    prev_best_save_ckpt.unlink()
            # Update best model
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_ckpt_basename = f"{run_name}-epoch_{best_epoch}-loss_{best_val_loss:.5e}.pt"
            torch.save(
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "sample_rate": sr_x,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                },
                save_dir / best_ckpt_basename,
            )

            # Save validation sample only for the current best model
            num_samples = min(val_audio_seconds * sr_x, x_val.shape[0], y_val.shape[0])
            sample_x = x_val[:num_samples].copy()
            sample_y = y_val[:num_samples].copy()
            with torch.no_grad():
                sample_in = torch.from_numpy(sample_x).to(device)[None, :]
                sample_pred = model(sample_in).squeeze(0).detach().cpu().numpy().astype(np.float32)
            sf.write(str(save_dir / "source.wav"), sample_x, sr_x)
            sf.write(str(save_dir / "target.wav"), sample_y, sr_x)
            sf.write(str(save_dir / "model_output.wav"), sample_pred, sr_x)

            log_message(log_path, "=============================================================")
            log_message(log_path, f"Updated best model - epoch={best_epoch} - val_loss={best_val_loss:.5e}")
            log_message(log_path, "=============================================================")
            
        epoch_loss_history.append(
            {
                "train_loss": float(avg_train_loss) if np.isfinite(avg_train_loss) else None,
                "val_loss": float(avg_val_loss) if np.isfinite(avg_val_loss) else None,
            }
        )
        with losses_json_path.open("w", encoding="utf-8") as f:
            json.dump(epoch_loss_history, f, indent=2)

    # Print final results
    final_sparsity = _global_sparsity(params_to_prune)
    total_prunable_params = sum(getattr(module, name).numel() for module, name in params_to_prune)
    total_params = sum(p.numel() for p in model.parameters())
    total_pruned_params = sum((getattr(module, name) == 0).sum().item() for module, name in params_to_prune)
    log_message(
        log_path,
        f"Final sparsity check:\n"
        f"  - Target sparsity: {sparsity_target:.6f}\n"
        f"  - Achieved sparsity: {final_sparsity:.6f}\n"
        f"  - Delta: {abs(final_sparsity - sparsity_target):.6f}\n"
        f"  - Total parameters: {total_params}\n"
        f"  - Total prunable parameters: {total_prunable_params}\n"
        f"  - Total parameters pruned: {total_pruned_params}\n"
        f"  - Sparsity (pruned/prunable): {total_pruned_params/total_prunable_params:.6f}\n"
        f"  - Actual sparsity (pruned/total): {total_pruned_params/total_params:.6f}"
    )
    if best_ckpt_basename is not None:
        log_message(log_path, f"Training complete. best_epoch={best_epoch} - best_val_loss={best_val_loss:.5e} - {save_dir / best_ckpt_basename}")
    else:
        log_message(log_path, "Training complete. No best checkpoint was saved.")
    elapsed_seconds = (datetime.now() - train_start_time).total_seconds()
    elapsed_h = int(elapsed_seconds // 3600)
    elapsed_m = int((elapsed_seconds % 3600) // 60)
    elapsed_s = elapsed_seconds % 60.0
    log_message(log_path, f"Total runtime: {elapsed_h:02d}:{elapsed_m:02d}:{elapsed_s:06.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple WaveNet training script.")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="model_cfg/example.json",
        help="Path to model config JSON (default: model_cfg/example.json).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        help="Batch size (default: 40).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="Number of epochs (default: 1500).",
    )
    parser.add_argument(
        "--input_wav",
        type=str,
        default="data/example/input.wav",
        help="Input wav path (default: data/example/input.wav).",
    )
    parser.add_argument(
        "--target_wav",
        type=str,
        default="data/example/output.wav",
        help="Target wav path (default: data/example/output.wav).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=4800,
        help="Chunk size in samples (default: 4800, i.e. 100 ms at 48 kHz).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Dataloader workers (default: 0).",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--val_audio_seconds",
        type=int,
        default=5,
        help="Validation audio export length in seconds (default: 5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device (default: cuda).",
    )
    parser.add_argument(
        "--sparsity_target",
        type=float,
        default=0.9,
        help="Target sparsity (default: 0.9).",
    )
    parser.add_argument(
        "--prune_type",
        type=str,
        default="local",
        help="Pruning type: global or local (default: local).",
    )
    parser.add_argument(
        "--prune_schedule",
        type=str,
        default="exponential",
        help="Pruning schedule: linear or exponential (default: exponential).",
    )
    parser.add_argument(
        "--prune_start_epoch",
        type=int,
        default=10,
        help="Epoch to start pruning (default: 10).",
    )
    parser.add_argument(
        "--prune_end_epoch",
        type=int,
        default=750,
        help="Epoch to end pruning (default: 750).",
    )
    args = parser.parse_args()
    model_cfg = load_json(args.model_cfg)
    train_cfg = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "input_wav": args.input_wav,
        "target_wav": args.target_wav,
        "chunk_size": args.chunk_size,
        "num_workers": args.num_workers,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "val_audio_seconds": args.val_audio_seconds,
        "device": args.device,
        "sparsity_target": args.sparsity_target,
        "prune_type": args.prune_type,
        "prune_schedule": args.prune_schedule,
        "prune_start_epoch": args.prune_start_epoch,
        "prune_end_epoch": args.prune_end_epoch,
    }
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
