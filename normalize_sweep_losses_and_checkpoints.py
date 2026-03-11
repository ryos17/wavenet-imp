import json
import re
from pathlib import Path


TRAIN_DIVISOR = 94.51431595651727
VAL_DIVISOR = 43.31156539916992

SWEEP_DIRS = [
    Path("models/prune_end_sweep"),
    Path("models/prune_type_schedule"),
    Path("models/sparsity_level_sweep"),
]

LOSS_PATTERN = re.compile(r"^(?P<prefix>.*-loss_)(?P<loss>[0-9eE+\-.]+)(?P<suffix>\.pt)$")


def normalize_losses_json(losses_path: Path) -> bool:
    with losses_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in {losses_path}, got {type(data).__name__}")

    changed = False
    for row in data:
        if not isinstance(row, dict):
            continue
        if "train_loss" in row and isinstance(row["train_loss"], (int, float)):
            row["train_loss"] = row["train_loss"] / TRAIN_DIVISOR
            changed = True
        if "val_loss" in row and isinstance(row["val_loss"], (int, float)):
            row["val_loss"] = row["val_loss"] / VAL_DIVISOR
            changed = True

    if changed:
        with losses_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    return changed


def rename_checkpoint_files(subdir: Path) -> int:
    renamed = 0
    for pt_path in subdir.glob("*.pt"):
        match = LOSS_PATTERN.match(pt_path.name)
        if not match:
            continue

        original_loss = float(match.group("loss"))
        scaled_loss = original_loss / VAL_DIVISOR
        new_name = f"{match.group('prefix')}{scaled_loss:.5e}{match.group('suffix')}"
        new_path = pt_path.with_name(new_name)

        if new_path == pt_path:
            continue
        if new_path.exists():
            raise FileExistsError(f"Cannot rename {pt_path} -> {new_path}: destination exists")

        pt_path.rename(new_path)
        renamed += 1

    return renamed


def process_subdirectory(subdir: Path) -> tuple[bool, int]:
    losses_path = subdir / "losses.json"
    losses_changed = False
    if losses_path.is_file():
        losses_changed = normalize_losses_json(losses_path)

    renamed_count = rename_checkpoint_files(subdir)
    return losses_changed, renamed_count


def main() -> None:
    root = Path(__file__).resolve().parent
    total_loss_files = 0
    total_renamed = 0

    for rel_parent in SWEEP_DIRS:
        parent = root / rel_parent
        if not parent.is_dir():
            print(f"[skip] missing directory: {parent}")
            continue

        for subdir in parent.iterdir():
            if not subdir.is_dir():
                continue

            losses_changed, renamed_count = process_subdirectory(subdir)
            if losses_changed:
                total_loss_files += 1
            total_renamed += renamed_count
            if losses_changed or renamed_count:
                print(
                    f"[ok] {subdir} | losses_updated={losses_changed} | "
                    f"checkpoints_renamed={renamed_count}"
                )

    print(
        f"Done. Updated {total_loss_files} losses.json files and renamed "
        f"{total_renamed} checkpoint files."
    )


if __name__ == "__main__":
    main()
