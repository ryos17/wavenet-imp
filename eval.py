from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from utils.util import read_audio_mono
from utils.wavenet import WaveNet


def load_model(model_path: str, device: torch.device) -> tuple[torch.nn.Module, int | None]:
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and "model_cfg" in ckpt:
        model = WaveNet.from_config_dict(ckpt["model_cfg"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        sample_rate = ckpt.get("sample_rate")
        return model.eval(), sample_rate

    if isinstance(ckpt, torch.nn.Module):
        return ckpt.to(device).eval(), None

    raise ValueError(
        "Unsupported model file format. Expected a checkpoint with "
        "'model_cfg' + 'model_state_dict' or a serialized torch.nn.Module."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Very simple WaveNet eval script.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model/checkpoint file.")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to input wav file.")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output wav path (default: outputs/{basename of input_wav}).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_wav)
    output_path = Path(args.output_path) if args.output_path else Path("outputs") / input_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_sr = load_model(args.model_path, device)

    x, sr = read_audio_mono(input_path)
    if checkpoint_sr is not None and checkpoint_sr != sr:
        print(f"Warning: checkpoint sample_rate={checkpoint_sr}, input sample_rate={sr}")

    with torch.no_grad():
        x_tensor = torch.from_numpy(x).to(device)[None, :]
        y_tensor = model(x_tensor)
        if y_tensor.ndim == 3 and y_tensor.shape[1] == 1:
            y_tensor = y_tensor[:, 0, :]
        y = y_tensor.squeeze(0).detach().cpu().numpy().astype("float32")

    sf.write(str(output_path), y, sr)
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
