from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch

from utils.util import (
    build_dataloader,
    load_json,
    log_message,
    pre_emphasis,
    read_audio_mono,
    set_seed,
)
from utils.wavenet import WaveNet


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

    # Set seed and device
    set_seed(seed)
    device = torch.device(device_name)

    # Create save directory with model config basename and date
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    wav_name = Path(target_wav).stem
    run_name = f"{wav_name}-b{batch_size}-lr{learning_rate}-e{epochs}-{run_stamp}"
    save_dir = Path("checkpoints") / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = save_dir / "logs.txt"
    losses_json_path = save_dir / "losses.json"
    log_message(log_path, f"Train config: {train_cfg}")
    log_message(log_path, f"Model config: {model_cfg}")

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

    # Train loop
    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_basename: str | None = None
    epoch_loss_history: list[dict[str, float | None]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
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
        log_message(log_path, f"Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.5e} - val_loss={avg_val_loss:.5e}")

        # Update best model and save validation artifacts
        if avg_val_loss < best_val_loss:
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
    if best_ckpt_basename is not None:
        log_message(log_path, f"Training complete. best_epoch={best_epoch} - best_val_loss={best_val_loss:.5e} - {save_dir / best_ckpt_basename}")
    else:
        log_message(log_path, "Training complete. No best checkpoint was saved.")


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
    }
    train(model_cfg, train_cfg)


if __name__ == "__main__":
    main()
