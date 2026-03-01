import argparse
from pprint import pprint

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a PyTorch model/checkpoint file and print its contents."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model/checkpoint file",
    )
    args = parser.parse_args()

    # Print model contents
    loaded_obj = torch.load(args.model_path, map_location="cpu")
    obj_without_state = {k: v for k, v in loaded_obj.items() if k != "model_state_dict"}
    pprint(obj_without_state)

    # Count pruning stats if pruning is present
    mask_items = [(k, v) for k, v in loaded_obj["model_state_dict"].items() if "weight_mask" in k]
    if mask_items:
        num_params = sum(mask.numel() for _, mask in mask_items)
        num_nonzero_params = sum((mask != 0).sum().item() for _, mask in mask_items)
        num_zero_params = num_params - num_nonzero_params
        total_params = sum(
            p.numel()
            for k, p in loaded_obj["model_state_dict"].items()
            if isinstance(p, torch.Tensor) and not k.endswith("weight_mask")
        )
        print(f"Total parameters: {total_params}")
        print(f"Total prunable parameters: {num_params}")
        print(f"Total pruned parameters: {num_zero_params}")
        print(f"Sparsity (by mask): {num_zero_params / num_params:.6f}")
        print(f"Actual sparsity (pruned/total): {num_zero_params / total_params:.6f}")


if __name__ == "__main__":
    main()
