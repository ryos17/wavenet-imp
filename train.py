from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utils.util import (
    build_dataloader,
    load_json,
    log_message,
    pre_emphasis,
    read_audio_mono,
    set_seed,
)
from utils.wavenet import WaveNet


def train(model_cfg: Dict, train_cfg: Dict, model_cfg_path: str, train_cfg_path: str) -> None:
    # Load train config
    seed = int(train_cfg["seed"])
    batch_size = int(train_cfg["batch_size"])
    learning_rate = float(train_cfg["learning_rate"])
    epochs = int(train_cfg["epochs"])
    input_wav = train_cfg["input_wav"]
    target_wav = train_cfg["target_wav"]
    chunk_size = int(train_cfg["chunk_size"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    val_steps_per_epoch = int(train_cfg["val_steps_per_epoch"])
    num_workers = int(train_cfg["num_workers"])
    train_split = float(train_cfg["train_split"])
    val_split = float(train_cfg["val_split"])
    val_audio_seconds = int(train_cfg["val_audio_seconds"])
    device_name = train_cfg["device"]

    # Set seed and device
    set_seed(seed)
    device = torch.device(device_name)

    # Create save directory
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    save_dir = Path("checkpoints") / run_stamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create log file
    log_path = save_dir / "logs.txt"
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
        steps_per_epoch=steps_per_epoch,
        num_workers=num_workers,
    )
    val_loader = build_dataloader(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        chunk_size=chunk_size,
        steps_per_epoch=val_steps_per_epoch,
        num_workers=num_workers,
    )

    # Train loop
    best_val_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=True)
        for xb, yb in pbar:
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

            # Update progress bar
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            pbar.set_postfix(loss=f"{loss_value:.6f}")

        # Compute average train loss
        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False)
            for xb, yb in val_pbar:
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

                # Update progress bar
                val_loss_value = float(val_loss.item())
                val_losses.append(val_loss_value)
                val_pbar.set_postfix(loss=f"{val_loss_value:.6f}")

        # Compute average validation loss
        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        # Create epoch directory and save checkpoint
        epoch_dir = save_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = epoch_dir / "checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
                "model_state_dict": model.state_dict(),
                "sample_rate": sr_x,
                "model_cfg_path": model_cfg_path,
                "train_cfg_path": train_cfg_path,
            },
            ckpt_path,
        )

        # Save validation sample
        num_samples = min(val_audio_seconds * sr_x, x_val.shape[0], y_val.shape[0])
        sample_x = x_val[:num_samples].copy()
        sample_y = y_val[:num_samples].copy()
        with torch.no_grad():
            sample_in = torch.from_numpy(sample_x).to(device)[None, :]
            sample_pred = model(sample_in).squeeze(0).detach().cpu().numpy().astype(np.float32)
        sf.write(str(epoch_dir / "source.wav"), sample_x, sr_x)
        sf.write(str(epoch_dir / "target.wav"), sample_y, sr_x)
        sf.write(str(epoch_dir / "model_output.wav"), sample_pred, sr_x)

        # Print epoch results
        log_message(
            log_path,
            f"Epoch {epoch}/{epochs} - train_loss={avg_train_loss:.5e} "
            f"- val_loss={avg_val_loss:.5e} - {epoch_dir}"
        )

        # Update best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "sample_rate": sr_x,
                    "model_cfg_path": model_cfg_path,
                    "train_cfg_path": train_cfg_path,
                },
                save_dir / "best.pt",
            )
            log_message(log_path, f"Updated best epoch - {save_dir / 'best.pt'}")

    # Print final results
    log_message(
        log_path,
        f"Training complete. best_epoch={best_epoch} - best_val_loss={best_val_loss:.5e} - {save_dir / 'best.pt'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple WaveNet training script.")
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="cfg/model/example.json",
        help="Path to model config JSON (default: cfg/model/example.json).",
    )
    parser.add_argument(
        "--train_cfg",
        type=str,
        default="cfg/train/example.json",
        help="Path to train config JSON (default: cfg/train/example.json).",
    )
    args = parser.parse_args()
    model_cfg = load_json(args.model_cfg)
    train_cfg = load_json(args.train_cfg)
    train(model_cfg, train_cfg, model_cfg_path=args.model_cfg, train_cfg_path=args.train_cfg)


if __name__ == "__main__":
    main()
