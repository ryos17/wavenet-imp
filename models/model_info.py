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

    loaded_obj = torch.load(args.model_path, map_location="cpu")
    pprint(loaded_obj)


if __name__ == "__main__":
    main()
