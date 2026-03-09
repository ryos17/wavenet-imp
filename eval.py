from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from utils.util import read_audio_mono
from utils.wavenet import WaveNet


def _materialize_pruned_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Convert *_orig/*_mask pruning keys into standard parameter keys."""
    converted: dict[str, torch.Tensor] = {}
    converted_count = 0

    # Keep regular parameters/buffers
    for key, value in state_dict.items():
        if key.endswith("_orig") or key.endswith("_mask"):
            continue
        converted[key] = value

    # Materialize pruned tensors: `weight = weight_orig * weight_mask`.
    for key, value in state_dict.items():
        if not key.endswith("_orig"):
            continue
        base_key = key[: -len("_orig")]
        mask_key = f"{base_key}_mask"
        converted[base_key] = value * state_dict[mask_key]
        converted_count += 1

    return converted, converted_count


def load_model(model_path: str, device: torch.device) -> tuple[torch.nn.Module, int | None]:
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)

    # Create model from config
    model = WaveNet.from_config_dict(ckpt["model_cfg"]).to(device)

    # Load model state dict, supporting pruning reparameterized checkpoints.
    state_dict, converted_count = _materialize_pruned_state_dict(ckpt["model_state_dict"])
    if converted_count > 0:
        print(f"Detected pruned checkpoint: materialized {converted_count} masked tensors.")
    model.load_state_dict(state_dict)

    # Return model and sample rate
    sample_rate = ckpt.get("sample_rate")
    return model.eval(), sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Very simple WaveNet eval script.")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input wav file.")
    parser.add_argument("--model_path", type=str, default="models/ch16_ungated-best.pt", help="Path to model/checkpoint file. (default: models/ch16_ungated-best.pt)")
    parser.add_argument("--output_path", type=str, default=None, help="Output wav path (default: outputs/{basename of input_wav}).")
    args = parser.parse_args()

    # Create output path
    input_path = Path(args.input_wav)
    output_path = Path(args.output_path) if args.output_path else Path("outputs") / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_sr = load_model(args.model_path, device)

    # Load input audio
    x, sr = read_audio_mono(input_path)

    # Check sample rate
    if checkpoint_sr is not None and checkpoint_sr != sr:
        print(f"Warning: checkpoint sample_rate={checkpoint_sr}, input sample_rate={sr}")

    # Generate output audio
    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device)[None, :]
        y_tensor = model(x_tensor)
        if y_tensor.ndim == 3 and y_tensor.shape[1] == 1:
            y_tensor = y_tensor[:, 0, :]
        y = y_tensor.squeeze(0).detach().cpu().numpy().astype("float32")

    # Save output audio
    sf.write(str(output_path), y, sr)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
